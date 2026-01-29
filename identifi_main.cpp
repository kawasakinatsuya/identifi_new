#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>

#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/filter.h>

#include <pcl/kdtree/kdtree.h>
#include <pcl/search/kdtree.h>
#include <pcl/segmentation/extract_clusters.h>

#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>

#include <deque>
#include <map>
#include <vector>
#include <limits>
#include <algorithm>
#include <cmath>
#include <unordered_set>
#include <Eigen/Dense>

#include <pcl/features/normal_3d.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/ModelCoefficients.h>

#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2/LinearMath/Vector3.h>

#include <mutex>
#include <cstdint>

// ============================================================
// Cell
// ============================================================
struct Cell
{
  int ix = 0;
  int iy = 0;

  std::vector<int> indices;  // cloud_map 内の点 index
  double min_z = std::numeric_limits<double>::infinity();  // セル全点の最小
  double z_max = -std::numeric_limits<double>::infinity(); // セル全点の最大

  // --- ground plane (地面クラスタ点のみ) ---
  std::vector<int> ground_indices;
  Eigen::Vector3f n_ground = Eigen::Vector3f(0,0,1); // 代表法線(新方針)
  float d_ground = 0.0f;                             // 代表平面d
  bool has_ground_plane = false;                     // 代表法線が取れたか

  // ground height stats (新方針)
  double ground_min_z = std::numeric_limits<double>::infinity();   // 地面高さ（最小）
  double ground_max_z = -std::numeric_limits<double>::infinity();  // 代表高さ（最大）

  // --- non-ground plane (非地面点のみ) ---
  std::vector<int> nonground_indices;
  Eigen::Vector3f n_ng = Eigen::Vector3f(0,0,1);
  float d_ng = 0.0f;
  bool has_ng_plane = false;

  // --- cell classification flags ---
  bool ground = false;    // 平地（緑）
  bool slope  = false;    // 斜面（シアン）
  bool small_obs = false; // 段差（黄）
  bool big_obs   = false; // 障害物（赤）
  bool unknown = false;

  bool cylinder = false;
};

static inline uint64_t pack2i(int a, int b)
{
  return (uint64_t)(uint32_t)a << 32 | (uint32_t)b;
}
static inline uint64_t cell_key_from_xy(double x, double y, double bx, double by, double cell_size)
{
  int ix = (int)std::floor((x - bx) / cell_size);
  int iy = (int)std::floor((y - by) / cell_size);
  return pack2i(ix, iy);
}


struct TimeCloud
{
  rclcpp::Time stamp;
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_world;
};

// ============================================================
// math helpers
// ============================================================
static inline double rad2deg(double rad)
{
  return rad * 180.0 / 3.14159265358979323846;
}

static inline double angle_deg(const Eigen::Vector3f& a, const Eigen::Vector3f& b)
{
  double c = (double)a.dot(b);
  c = std::max(-1.0, std::min(1.0, c));
  return rad2deg(std::acos(c));
}

// 平面 n.x*x + n.y*y + n.z*z + d = 0 から z を求める
static inline bool plane_z_at_xy(const Eigen::Vector3f& n, float d, float x, float y, float& z_out)
{
  if(std::abs(n.z()) < 1e-6f) return false;
  z_out = -(n.x()*x + n.y()*y + d) / n.z();
  return std::isfinite(z_out);
}

// PCAで平面を当てて法線を求める
static inline bool fit_plane_pca(
  const pcl::PointCloud<pcl::PointXYZ>& cloud,
  const std::vector<int>& indices,
  Eigen::Vector3f& n_out,
  float& d_out)
{
  if ((int)indices.size() < 10) return false;

  Eigen::Vector3f mean(0,0,0);
  int cnt = 0;
  for(int idx : indices){
    const auto& p = cloud.points[idx];
    if(!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z)) continue;
    mean += Eigen::Vector3f(p.x, p.y, p.z);
    cnt++;
  }
  if(cnt < 10) return false;
  mean /= (float)cnt;

  Eigen::Matrix3f cov = Eigen::Matrix3f::Zero();
  for(int idx : indices){
    const auto& p = cloud.points[idx];
    if(!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z)) continue;
    Eigen::Vector3f q(p.x, p.y, p.z);
    Eigen::Vector3f d = q - mean;
    cov += d * d.transpose();
  }
  cov /= (float)cnt;

  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> es(cov);
  if(es.info() != Eigen::Success) return false;

  // 最小固有値の固有ベクトルが法線
  Eigen::Vector3f n = es.eigenvectors().col(0);
  n.normalize();

  // 上向きに揃える
  if(n.z() < 0) n = -n;

  n_out = n;
  d_out = -n.dot(mean);
  return true;
}

// ground_indices の「中央値高さの点」を返す（cloud_map の index）
static inline int select_median_z_index(
  const pcl::PointCloud<pcl::PointXYZ>& cloud,
  const std::vector<int>& indices)
{
  std::vector<std::pair<float,int>> vz;
  vz.reserve(indices.size());
  for(int idx: indices){
    const auto& p = cloud.points[idx];
    if(!std::isfinite(p.z)) continue;
    vz.push_back({p.z, idx});
  }
  if(vz.size() < 3) return -1;

  auto mid_it = vz.begin() + (ptrdiff_t)(vz.size()/2);
  std::nth_element(vz.begin(), mid_it, vz.end(),
                   [](const auto& a, const auto& b){ return a.first < b.first; });
  return mid_it->second;
}

// 代表点の周囲から ground_set の点だけを集めて PCA 法線
static inline bool fit_rep_normal_knn30(
  const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud_map,
  pcl::search::KdTree<pcl::PointXYZ>::Ptr& kdtree_map,
  int rep_idx,                                 // cloud_map index
  const std::unordered_set<int>& ground_set,   // ground indices in cloud_map
  Eigen::Vector3f& n_out, float& d_out)
{
  if(rep_idx < 0) return false;
  const auto& pref = cloud_map->points[rep_idx];
  if(!std::isfinite(pref.x)||!std::isfinite(pref.y)||!std::isfinite(pref.z)) return false;

  std::vector<int> nn_idx;
  std::vector<float> nn_d2;

  // 多めに取って、ground_set に入ってるものだけ 30 個使う
  if(kdtree_map->nearestKSearch(pref, 120, nn_idx, nn_d2) <= 0) return false;

  std::vector<int> use;
  use.reserve(35);
  for(int j : nn_idx){
    if((int)use.size() >= 30) break;
    if(ground_set.find(j) == ground_set.end()) continue;
    use.push_back(j);
  }
  if((int)use.size() < 10) return false;

  if(!fit_plane_pca(*cloud_map, use, n_out, d_out)) return false;

  // 代表点を通るように d を調整（局所平面）
  Eigen::Vector3f p(pref.x, pref.y, pref.z);
  d_out = -n_out.dot(p);
  return true;
}

// ============================================================
// Node
// ============================================================
class IdentificationNode : public rclcpp::Node
{
public:
  IdentificationNode()
  : Node("Identification_node"), tf_buffer_(this->get_clock()), tf_listener_(tf_buffer_)
  {
    sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
      "/cloud_registered", rclcpp::SensorDataQoS(),
      std::bind(&IdentificationNode::cloud_callback, this, std::placeholders::_1));

    sub_weed_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
      "/weed_points", rclcpp::SensorDataQoS(),
      std::bind(&IdentificationNode::weed_callback, this, std::placeholders::_1));

    pub_colored_      = this->create_publisher<sensor_msgs::msg::PointCloud2>("/points_colored", 10);
    pub_ground_       = this->create_publisher<sensor_msgs::msg::PointCloud2>("/ground", 10);
    pub_unknown_      = this->create_publisher<sensor_msgs::msg::PointCloud2>("/unknown", 10);
    pub_cylinder_     = this->create_publisher<sensor_msgs::msg::PointCloud2>("/cylinder", 10);
    pub_cell_ground_  = this->create_publisher<sensor_msgs::msg::PointCloud2>("/cell_ground", 10);

    // frames
    this->declare_parameter("world_frame", "camera_init");
    this->declare_parameter("base_frame",  "body");

    // map build
    this->declare_parameter("voxel_size", 0.03);
    this->declare_parameter("time_window", 10.0);

    // grid
    this->declare_parameter("cell_size", 0.10);
    this->declare_parameter("min_points_per_cell", 3);

    // Step1: “最下点 + 帯域”
    this->declare_parameter("ground_band", 0.10);

    // Step2: clustering parameters
    this->declare_parameter("cluster_tolerance", 0.20);
    this->declare_parameter("min_cluster_size", 200);
    this->declare_parameter("max_cluster_size", 200000);

    this->declare_parameter("extra_ground_z_margin", 0.05);
    this->declare_parameter("extra_ground_size_ratio", 0.10);

    // ---- 識別条件 ----
    this->declare_parameter("flat_deg_max", 12.0);        // 平地とみなす最大角度
    this->declare_parameter("slope_deg_max", 30.0);       // 斜面とみなす最大角度
    this->declare_parameter("step_h_max", 0.15);          // 非地面分類の 高さ差 しきい値（段差/障害物）

    this->declare_parameter("ref_search_radius", 1.0);    // [m]
    this->declare_parameter("flat_ref_deg_max", 10.0);    // [deg] 近傍基準地面探索の“保険”で使う

    this->declare_parameter("ref_search_cell", 2);        // 周囲何セルを比較対象にするか

    // ---- 新方針：地面クラスタ同士の比較しきい値 ----
    this->declare_parameter("step_ang_deg", 15.0);        // 代表法線差
    this->declare_parameter("step_h_rep", 0.10);          // 代表高さ差（ground_max_z 同士）

    // cylinder
    this->declare_parameter("cylinder_enable", true);
    this->declare_parameter("cylinder_roi_radius", 0.25);
    this->declare_parameter("cylinder_roi_z_min", -1.0);
    this->declare_parameter("cylinder_roi_z_max",  2.0);
    this->declare_parameter("cylinder_min_roi_points", 80);
    this->declare_parameter("cylinder_normal_k", 20);
    this->declare_parameter("cylinder_dist_thresh", 0.05);
    this->declare_parameter("cylinder_radius_min", 0.03);
    this->declare_parameter("cylinder_radius_max", 0.25);
    this->declare_parameter("cylinder_min_inliers", 60);
    this->declare_parameter("cylinder_max_cells_per_frame", 60);

    // weed
    this->declare_parameter("weed_cell_size", 0.10); // [m]
    this->declare_parameter("weed_zrel_min", 0.03);

    // roi
    this->declare_parameter("roi_enable", true);
    this->declare_parameter("roi_x_min", 0.0);   // [m]
    this->declare_parameter("roi_x_max", 3.0);   // [m]
    this->declare_parameter("roi_y_max", 0.90);
    this->declare_parameter("roi_y_min", -1.5);

    this->declare_parameter("weed_radius", 0.10); // [m]

    RCLCPP_INFO(this->get_logger(),
      "IdentificationNode initialized (min-z per cell + largest cluster + NEW slope/step policy + cylinder/weed kept).");
  }

private:
  // ==========================================================
  // callback
  // ==========================================================
  void cloud_callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
  {
    const std::string world_frame = this->get_parameter("world_frame").as_string();
    const std::string base_frame  = this->get_parameter("base_frame").as_string();

    // msg -> pcl
    pcl::PointCloud<pcl::PointXYZ>::Ptr input(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::fromROSMsg(*msg, *input);

    // remove NaN
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZ>);
    std::vector<int> nan_indices;
    pcl::removeNaNFromPointCloud(*input, *cloud_filtered, nan_indices);

    const rclcpp::Time stamp(msg->header.stamp);

    // TF lookup (world <- base)
    geometry_msgs::msg::TransformStamped tf_w_b;
    try{
      tf_w_b = tf_buffer_.lookupTransform(world_frame, base_frame, stamp);
    }catch(const std::exception &e){
      RCLCPP_WARN(this->get_logger(), "TF lookup failed (%s -> %s): %s",
                  base_frame.c_str(), world_frame.c_str(), e.what());
      return;
    }

    const double bx = tf_w_b.transform.translation.x;
    const double by = tf_w_b.transform.translation.y;

    // ROI filter in base
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_roi = cloud_filtered;
    const bool roi_enable = this->get_parameter("roi_enable").as_bool();
    if(roi_enable){
      const float roi_x_min = (float)this->get_parameter("roi_x_min").as_double();
      const float roi_x_max = (float)this->get_parameter("roi_x_max").as_double();
      const float roi_y_max = (float)this->get_parameter("roi_y_max").as_double();
      const float roi_y_min = (float)this->get_parameter("roi_y_min").as_double();
      cloud_roi = roi_filter_in_base(cloud_filtered, tf_w_b, roi_x_min, roi_x_max, roi_y_min, roi_y_max);
    }

    // downsample current frame
    const double voxel_size = this->get_parameter("voxel_size").as_double();
    pcl::PointCloud<pcl::PointXYZ>::Ptr down = voxel_downsample(cloud_roi, voxel_size);

    // push to buffer
    TimeCloud tc;
    tc.stamp = stamp;
    tc.cloud_world = down;
    cloud_buffer_.push_back(tc);

    // pop old
    const double time_window = this->get_parameter("time_window").as_double();
    pop_old_clouds(stamp, time_window);

    // build merged map and downsample again
    pcl::PointCloud<pcl::PointXYZ>::Ptr merged = build_map();
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_map = voxel_downsample(merged, voxel_size);

    // Step1: build grid and min_z per cell
    const double cell_size = this->get_parameter("cell_size").as_double();
    const int min_pts_cell = this->get_parameter("min_points_per_cell").as_int();

    std::map<std::pair<int,int>, Cell> cells;
    build_grid_and_minz(*cloud_map, cells, bx, by, cell_size);

    // collect ground-candidate indices (min_z + band)
    const double ground_band = this->get_parameter("ground_band").as_double();
    std::vector<int> candidate_indices;
    candidate_indices.reserve(cloud_map->points.size() / 4);

    for(const auto& kv : cells){
      const Cell& c = kv.second;
      if((int)c.indices.size() < min_pts_cell) continue;
      if(!std::isfinite(c.min_z)) continue;

      const double z_thr = c.min_z + ground_band;

      for(int idx : c.indices){
        const auto& pt = cloud_map->points[idx];
        if(!std::isfinite(pt.z)) continue;
        if((double)pt.z <= z_thr){
          candidate_indices.push_back(idx);
        }
      }
    }

    if(candidate_indices.size() < 100){
      publish_all_nonground(*msg, cloud_map);
      return;
    }

    // unique
    std::sort(candidate_indices.begin(), candidate_indices.end());
    candidate_indices.erase(std::unique(candidate_indices.begin(), candidate_indices.end()), candidate_indices.end());

    // build candidate cloud
    pcl::PointCloud<pcl::PointXYZ>::Ptr cand_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    cand_cloud->points.reserve(candidate_indices.size());

    // map: cand_cloud index -> original index
    std::vector<int> cand_to_orig;
    cand_to_orig.reserve(candidate_indices.size());

    for(int orig_idx : candidate_indices){
      cand_cloud->points.push_back(cloud_map->points[orig_idx]);
      cand_to_orig.push_back(orig_idx);
    }
    cand_cloud->width = (uint32_t)cand_cloud->points.size();
    cand_cloud->height = 1;

    // Step2: Euclidean clustering on candidate points
    std::vector<pcl::PointIndices> cluster_indices;
    euclidean_cluster(cand_cloud, cluster_indices);

    if(cluster_indices.empty()){
      publish_all_nonground(*msg, cloud_map);
      return;
    }

    // pick largest cluster
    int best_k = -1;
    size_t best_size = 0;
    for(int k = 0; k < (int)cluster_indices.size(); ++k){
      size_t sz = cluster_indices[k].indices.size();
      if(sz > best_size){
        best_size = sz;
        best_k = k;
      }
    }

    if(best_k < 0 || best_size < 50){
      publish_all_nonground(*msg, cloud_map);
      return;
    }

    const double z_margin  = this->get_parameter("extra_ground_z_margin").as_double();
    const double ratio_min = this->get_parameter("extra_ground_size_ratio").as_double();

    double best_min_z = cluster_min_z(cand_cloud, cluster_indices, best_k);
    if(!std::isfinite(best_min_z)){
      publish_all_nonground(*msg, cloud_map);
      return;
    }

    std::vector<int> ground_cluster_ids;
    ground_cluster_ids.push_back(best_k);
    const size_t best_sz = cluster_indices[best_k].indices.size();

    for(int k = 0; k < (int)cluster_indices.size(); ++k){
      if(k == best_k) continue;
      const size_t sz = cluster_indices[k].indices.size();
      if(sz < (size_t)std::ceil((double)best_sz * ratio_min)) continue;
      double mz = cluster_min_z(cand_cloud, cluster_indices, k);
      if(!std::isfinite(mz)) continue;
      if(mz <= best_min_z + z_margin){
        ground_cluster_ids.push_back(k);
      }
    }

    // build ground mask (original indices in cloud_map)
    std::unordered_set<int> ground_set;
    ground_set.reserve(best_size * 2);
    for(int cid : ground_cluster_ids){
      for(int cand_idx : cluster_indices[cid].indices){
        int orig_idx = cand_to_orig[cand_idx];
        ground_set.insert(orig_idx);
      }
    }
        
    for(const auto& kv : cells){
        const Cell& c = kv.second;
        
        bool has_g  = false;
        bool has_ng = false;
        for(int idx : c.indices){
            if(ground_set.find(idx) != ground_set.end()) has_g = true;
            else has_ng = true;
            if(has_g && has_ng) break;
        }
        if(!(has_g && has_ng)) continue;

        // 混在セル：このセル内の ground_set点を除去
        for(int idx : c.indices){
            auto it = ground_set.find(idx);
            if(it != ground_set.end()){
                ground_set.erase(it);
            }
        }
    }

    // ==========================================================
    // NEW POLICY PART
    //   1) ground cluster points -> rep normal (median z point's knn30) + rep height(max) + ground height(min)
    //   2) ground-vs-neighbors -> ground/slope/small_obs
    //   3) non-ground plane -> ground/slope/small_obs/big_obs by angle + height diff (z_max - ground_min_z)
    // ==========================================================

    // KDTree for rep normal knn
    pcl::search::KdTree<pcl::PointXYZ>::Ptr kdtree_map(new pcl::search::KdTree<pcl::PointXYZ>);
    kdtree_map->setInputCloud(cloud_map);

    // 1) fill indices + compute rep for ground + pca for nonground
    for(auto& kv : cells){
      Cell& c = kv.second;

      c.ground_indices.clear();
      c.nonground_indices.clear();
      c.has_ground_plane = false;

      c.has_ng_plane = false;

      c.ground_min_z = std::numeric_limits<double>::infinity();
      c.ground_max_z = -std::numeric_limits<double>::infinity();

      c.ground = c.slope = c.small_obs = c.big_obs = c.unknown = false;

      for(int idx : c.indices){
        if(ground_set.find(idx) != ground_set.end()){
          c.ground_indices.push_back(idx);
          const auto& p = cloud_map->points[idx];
          if(std::isfinite(p.z)){
            c.ground_min_z = std::min(c.ground_min_z, (double)p.z);
            c.ground_max_z = std::max(c.ground_max_z, (double)p.z);
          }
        }else{
          c.nonground_indices.push_back(idx);
        }
      }

      // --- ground rep normal (median height point -> knn30 on ground_set) ---
      if((int)c.ground_indices.size() >= min_pts_cell){
        int rep_idx = select_median_z_index(*cloud_map, c.ground_indices);

        Eigen::Vector3f n;
        float d;
        bool ok = fit_rep_normal_knn30(cloud_map, kdtree_map, rep_idx, ground_set, n, d);

        if(!ok){
          // fallback: PCA on cell ground points
          ok = fit_plane_pca(*cloud_map, c.ground_indices, n, d);
          if(ok){
            // pass through rep point if possible
            int use_idx = (rep_idx >= 0 ? rep_idx : c.ground_indices.front());
            const auto& pr = cloud_map->points[use_idx];
            if(std::isfinite(pr.x)&&std::isfinite(pr.y)&&std::isfinite(pr.z)){
              Eigen::Vector3f p(pr.x, pr.y, pr.z);
              d = -n.dot(p);
            }
          }
        }

        if(ok){
          c.n_ground = n;
          c.d_ground = d;
          c.has_ground_plane = true;
        }
      }

      // --- non-ground plane (as before) ---
      if((int)c.nonground_indices.size() >= min_pts_cell){
        Eigen::Vector3f n;
        float d;
        if(fit_plane_pca(*cloud_map, c.nonground_indices, n, d)){
          c.n_ng = n;
          c.d_ng = d;
          c.has_ng_plane = true;
        }
      }
    }

    // ---- params ----
    const double flat_deg_max     = this->get_parameter("flat_deg_max").as_double();
    const double slope_deg_max    = this->get_parameter("slope_deg_max").as_double();
    const double step_h_max       = this->get_parameter("step_h_max").as_double();
    const double ref_R            = this->get_parameter("ref_search_radius").as_double();
    const double flat_ref_deg_max = this->get_parameter("flat_ref_deg_max").as_double();

    const int ref_search_cell = this->get_parameter("ref_search_cell").as_int();

    const double step_ang_deg     = this->get_parameter("step_ang_deg").as_double();
    const double step_h_rep       = this->get_parameter("step_h_rep").as_double();

    Eigen::Vector3f z_axis(0,0,1);

    // 2) ground cluster compare among neighbors:
    //    - both over -> small_obs
    //    - height only over -> slope
    //    - neither -> ground
    //    - angle-only over -> slope (実用上ここに寄せる)
    for(auto& kv : cells){
      Cell& c = kv.second;

      c.ground = c.slope = c.small_obs = c.big_obs = c.unknown = false;

      if(!c.has_ground_plane || !std::isfinite(c.ground_max_z)){
        c.unknown = true;
        continue;
      }

      double max_dh = 0.0;
      double max_da = 0.0;
      bool has_nb = false;

      for(int dy=-ref_search_cell; dy<=ref_search_cell; ++dy){
        for(int dx=-ref_search_cell; dx<=ref_search_cell; ++dx){
          if(dx==0 && dy==0) continue;
          auto it = cells.find({c.ix + dx, c.iy + dy});
          if(it == cells.end()) continue;

          const Cell& nb = it->second;
          if(!nb.has_ground_plane) continue;
          if(!std::isfinite(nb.ground_max_z)) continue;

          has_nb = true;

          double dh = std::abs(c.ground_max_z - nb.ground_max_z);
          double da = angle_deg(c.n_ground, nb.n_ground);

          if(dh > max_dh) max_dh = dh;
          if(da > max_da) max_da = da;
        }
      }

      if(!has_nb){
        c.ground = true;
        continue;
      }

      const bool h_over = (max_dh > step_h_rep);
      const bool a_over = (max_da > step_ang_deg);
      const double tilt_self = angle_deg(c.n_ground, z_axis);

      if(h_over && a_over){
        c.small_obs = true;
      }
      else if(tilt_self >= flat_deg_max){
        c.slope = true; 
      }
      else{
        c.ground = true;
      }
    }

    // 3) non-ground plane classification (overwrite by your rule)
    for(auto& kv : cells){
      Cell& c = kv.second;

      if(!c.has_ng_plane) continue;

      const double tilt = angle_deg(c.n_ng, z_axis);

      // ground height (prefer ground_min_z, else try neighbor plane at center)
      double gz = std::numeric_limits<double>::quiet_NaN();
      if(std::isfinite(c.ground_min_z)){
        gz = c.ground_min_z;
      }else{
        float xc = (float)(bx + (c.ix + 0.5) * cell_size);
        float yc = (float)(by + (c.iy + 0.5) * cell_size);
        float gz_f;
        if(lookup_ground_z_from_cells((double)xc, (double)yc, cells, bx, by, cell_size, ref_R, flat_ref_deg_max, gz_f)){
          gz = (double)gz_f;
        }
      }
      if(!std::isfinite(gz)) continue;

      double dz = 0.0;
      if(std::isfinite(c.z_max)) dz = c.z_max - gz;

      // ---- your rule ----
      if(tilt <= flat_deg_max){
        c.ground = true;
        c.slope = c.small_obs = c.big_obs = c.unknown = false;
      }
      else if(tilt <= slope_deg_max){
        c.slope = true;
        c.ground = c.small_obs = c.big_obs = c.unknown = false;
      }
      else{
        // steep
        if(dz >= step_h_max){
          c.big_obs = true;
          c.ground = c.slope = c.small_obs = c.unknown = false;
        }else{
          c.small_obs = true;
          c.ground = c.slope = c.big_obs = c.unknown = false;
        }
      }
    }

    // ==========================================================
    // cylinder detection (unchanged)
    // ==========================================================
    const bool cylinder_enable = this->get_parameter("cylinder_enable").as_bool();
    if(cylinder_enable){
      const double roi_R = this->get_parameter("cylinder_roi_radius").as_double();
      const double roi_zmin = this->get_parameter("cylinder_roi_z_min").as_double();
      const double roi_zmax = this->get_parameter("cylinder_roi_z_max").as_double();
      const int min_roi_pts = this->get_parameter("cylinder_min_roi_points").as_int();
      const int normal_k = this->get_parameter("cylinder_normal_k").as_int();
      const double dist_th = this->get_parameter("cylinder_dist_thresh").as_double();
      const double r_min = this->get_parameter("cylinder_radius_min").as_double();
      const double r_max = this->get_parameter("cylinder_radius_max").as_double();
      const int min_inliers   = this->get_parameter("cylinder_min_inliers").as_int();
      const int max_cells     = this->get_parameter("cylinder_max_cells_per_frame").as_int();

      const float front_min_x = 0.0f;
      const float front_max_x = 2.0f;
      const float side_max_y  = 0.5f;

      pcl::search::KdTree<pcl::PointXYZ>::Ptr kdtree(new pcl::search::KdTree<pcl::PointXYZ>);
      kdtree->setInputCloud(cloud_map);

      for(auto& kv : cells){
        kv.second.cylinder = false;
      }

      tf2::Quaternion q(tf_w_b.transform.rotation.x, tf_w_b.transform.rotation.y,
                        tf_w_b.transform.rotation.z, tf_w_b.transform.rotation.w);
      tf2::Matrix3x3 R(q);

      int processed = 0;
      int detected  = 0;

      for(auto& kv : cells){
        if(processed >= max_cells) break;
        Cell& c = kv.second;

        if(!(c.big_obs || c.small_obs)) continue;

        float xc = (float)(bx + (c.ix + 0.5) * cell_size);
        float yc = (float)(by + (c.iy + 0.5) * cell_size);

        // セル中心を base座標に変換して前方矩形でフィルタ
        tf2::Vector3 pw(xc - (float)bx, yc - (float)by, 0.0f);
        tf2::Vector3 pb = R.transpose() * pw;

        float x_base = (float)pb.x();
        float y_base = (float)pb.y();
        if(x_base < front_min_x) continue;
        if(x_base > front_max_x) continue;
        if(std::abs(y_base) > side_max_y) continue;

        processed++;

        bool ok = detect_cylinder_in_roi(
          cloud_map, kdtree, xc, yc,
          (float)roi_R, (float)roi_zmin, (float)roi_zmax,
          min_roi_pts, normal_k, (float)dist_th, (float)r_min, (float)r_max,
          min_inliers);

        if(ok){
          detected++;
          c.cylinder = true;
        }
      }

      RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
                           "cylinder: processed=%d detected=%d", processed, detected);
    }

    // ==========================================================
    // publish clouds (unchanged)
    // ==========================================================
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr ground(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr unknown(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cylinder(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr colored(new pcl::PointCloud<pcl::PointXYZRGB>);

    ground->points.reserve(cloud_map->points.size());
    unknown->points.reserve(cloud_map->points.size());
    colored->points.reserve(cloud_map->points.size());
    cylinder->points.reserve(cloud_map->points.size());

    auto cell_key_of_point = [&](const pcl::PointXYZ& pt)->std::pair<int,int>{
      double xl = (double)pt.x - bx;
      double yl = (double)pt.y - by;
      int ix = (int)std::floor(xl / cell_size);
      int iy = (int)std::floor(yl / cell_size);
      return {ix, iy};
    };

    pcl::PointCloud<pcl::PointXYZ>::Ptr weed_cloud_local(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::search::KdTree<pcl::PointXYZ>::Ptr weed_kdtree_local(new pcl::search::KdTree<pcl::PointXYZ>);
    bool has_weed = false;

    {
      std::lock_guard<std::mutex> lk(weed_mtx_);
      has_weed = has_weed_;
      if(has_weed){
        *weed_cloud_local = *weed_cloud_;          // ← weed_callbackで保持してる点群
        weed_kdtree_local->setInputCloud(weed_cloud_local);
      }
    }


    const double weed_radius = this->get_parameter("weed_radius").as_double();
    const double weed_zrel_min = this->get_parameter("weed_zrel_min").as_double();

    std::unordered_set<uint64_t> weed_cell_keys;
    weed_cell_keys.reserve(2048);
    
    if(has_weed){
      for(const auto& wp : weed_cloud_local->points){
        if(!std::isfinite(wp.x) || !std::isfinite(wp.y) || !std::isfinite(wp.z)) continue;

        // weed点が「地面から浮いてる」ものだけセルマーク（地面点が混ざる場合の保険）
        float gz;
        bool ok_gz = lookup_ground_z_from_cells(
          (double)wp.x, (double)wp.y, cells, bx, by, cell_size, ref_R, flat_ref_deg_max, gz);

        if(ok_gz){
          double zrel = (double)wp.z - (double)gz;
          if(zrel > weed_zrel_min){
            weed_cell_keys.insert(cell_key_from_xy((double)wp.x, (double)wp.y, bx, by, cell_size));
          }
        }else{
          // gzが取れないときは、要件優先でとりあえずセルマーク（必要なら消してOK）
          weed_cell_keys.insert(cell_key_from_xy((double)wp.x, (double)wp.y, bx, by, cell_size));
        }
      }
    }
    
    for(int i = 0; i < (int)cloud_map->points.size(); ++i){
      const auto& pt = cloud_map->points[i];

      pcl::PointXYZRGB p_ground, p_cyl, p_all;
      p_ground.x = p_cyl.x = p_all.x = pt.x;
      p_ground.y = p_cyl.y = p_all.y = pt.y;
      p_ground.z = p_cyl.z = p_all.z = pt.z;

      auto key = cell_key_of_point(pt);
      auto it  = cells.find(key);

      bool is_cylinder = false;
      bool is_ground = false;
      bool is_slope = false;
      bool is_small_obs = false;
      bool is_big_obs = false;

      bool is_ground_point = (ground_set.find(i) != ground_set.end());

      bool is_weed_near = false;
      if(has_weed){
        pcl::PointXYZ q;
        q.x = pt.x; q.y = pt.y; q.z = pt.z;

        std::vector<int> idx;
        std::vector<float> d2;
        if(weed_kdtree_local->radiusSearch(q, weed_radius, idx, d2) > 0){
          is_weed_near = true;
        }
      }

      bool is_weed_cell = false;
      if(!weed_cell_keys.empty()){
        uint64_t ck = cell_key_from_xy((double)pt.x, (double)pt.y, bx, by, cell_size);
        is_weed_cell = (weed_cell_keys.find(ck) != weed_cell_keys.end());
      }

      // NEW: 最終weed候補（半径10cm OR weedセル）
      bool weed_candidate = (is_weed_near || is_weed_cell);

      // ---- 地面高さ gz を近傍セルから借りて計算 -> zrel ----
      bool weed_ok_height = false;
      if(weed_candidate){
        float gz;
        bool ok_gz = lookup_ground_z_from_cells(
          (double)pt.x, (double)pt.y, cells, bx, by, cell_size, ref_R, flat_ref_deg_max, gz);

        if(ok_gz){
          double zrel = (double)pt.z - (double)gz;
          weed_ok_height = (zrel > weed_zrel_min);
        }
      }

      if(it == cells.end()){
        unknown->points.push_back(pt);
        p_all.r = 255; p_all.g = 0; p_all.b = 255;
        colored->points.push_back(p_all);
        continue;
      }else{
        const Cell& c = it->second;
        is_cylinder  = c.cylinder;
        is_ground    = c.ground;
        is_slope     = c.slope;
        is_small_obs = c.small_obs;
        is_big_obs   = c.big_obs;
      }

      // --- colorize all points ---
      if(is_ground){
        p_all.r = 0; p_all.g = 255; p_all.b = 0;         // ground green
      }
      else if(weed_candidate && weed_ok_height && !is_ground_point){
        p_all.r = 0; p_all.g = 0; p_all.b = 255;         // weed blue
      }
      else if(is_cylinder){
        p_all.r = 255; p_all.g = 0; p_all.b = 0;     // cylinder pink
      }
      else if(is_slope){
        p_all.r = 255; p_all.g = 0; p_all.b = 255;       // slope cyan
      }
      else if(is_small_obs){
        p_all.r = 255; p_all.g = 140; p_all.b = 0;       // step yellow
      }
      else if(is_big_obs){
        p_all.r = 255; p_all.g = 0; p_all.b = 0;         // obstacle red
      }
      else{
        p_all.r = 255; p_all.g = 255; p_all.b = 255;       // unknown purple
        unknown->points.push_back(pt);
      }
      colored->points.push_back(p_all);

      // ground_set points -> red, else white
      if(is_ground_point){
        p_ground.r = 255; p_ground.g = 0;   p_ground.b = 0;
      }else{
        p_ground.r = 255; p_ground.g = 255; p_ground.b = 255;
      }
      ground->points.push_back(p_ground);

      // cylinder points -> red, else white
      if(is_cylinder){
        p_cyl.r = 255; p_cyl.g = 0; p_cyl.b = 0;
      }else{
        p_cyl.r = 255; p_cyl.g = 255; p_cyl.b = 255;
      }
      cylinder->points.push_back(p_cyl);
    }

    // cell_ground: representative ground plane z at cell center (use n_ground/d_ground)
    pcl::PointCloud<pcl::PointXYZ>::Ptr cell_ground_cloud(new pcl::PointCloud<pcl::PointXYZ>());
    cell_ground_cloud->points.reserve(cells.size());

    for(const auto &kv : cells){
      const Cell &c = kv.second;
      if(!c.has_ground_plane) continue;

      float xc_w = (float)(bx + (c.ix + 0.5) * cell_size);
      float yc_w = (float)(by + (c.iy + 0.5) * cell_size);
      float zc_w = 0.f;

      if(!plane_z_at_xy(c.n_ground, c.d_ground, xc_w, yc_w, zc_w)) continue;

      pcl::PointXYZ pg;
      pg.x = xc_w;
      pg.y = yc_w;
      pg.z = zc_w;

      if(std::isfinite(pg.x) && std::isfinite(pg.y) && std::isfinite(pg.z)){
        cell_ground_cloud->points.push_back(pg);
      }
    }

    cell_ground_cloud->width  = (uint32_t)cell_ground_cloud->points.size();
    cell_ground_cloud->height = 1;
    cell_ground_cloud->is_dense = true;

    sensor_msgs::msg::PointCloud2 msg_cell_ground;
    pcl::toROSMsg(*cell_ground_cloud, msg_cell_ground);
    msg_cell_ground.header = msg->header;
    msg_cell_ground.header.frame_id = world_frame;
    pub_cell_ground_->publish(msg_cell_ground);

    finalize_cloud(ground);
    finalize_cloud(unknown);
    finalize_cloud(colored);
    finalize_cloud(cylinder);

    sensor_msgs::msg::PointCloud2 msg_col, msg_g, msg_ng, msg_cy;
    pcl::toROSMsg(*colored, msg_col);
    pcl::toROSMsg(*ground, msg_g);
    pcl::toROSMsg(*unknown, msg_ng);
    pcl::toROSMsg(*cylinder, msg_cy);

    msg_col.header = msg->header;
    msg_g.header   = msg->header;
    msg_ng.header  = msg->header;
    msg_cy.header  = msg->header;

    pub_colored_->publish(msg_col);
    pub_ground_->publish(msg_g);
    pub_unknown_->publish(msg_ng);
    pub_cylinder_->publish(msg_cy);

    RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
      "map=%zu cand=%zu clusters=%zu ground_set=%zu out_ground=%zu",
      cloud_map->points.size(),
      cand_cloud->points.size(),
      cluster_indices.size(),
      ground_set.size(),
      ground->points.size());
  }

  // ==========================================================
  // helpers
  // ==========================================================
  pcl::PointCloud<pcl::PointXYZ>::Ptr voxel_downsample(const pcl::PointCloud<pcl::PointXYZ>::Ptr& input, double vx)
  {
    pcl::VoxelGrid<pcl::PointXYZ> voxel;
    voxel.setInputCloud(input);
    voxel.setLeafSize((float)vx, (float)vx, (float)vx);

    pcl::PointCloud<pcl::PointXYZ>::Ptr out(new pcl::PointCloud<pcl::PointXYZ>);
    voxel.filter(*out);
    return out;
  }

  void pop_old_clouds(const rclcpp::Time& now, double time_window)
  {
    while(!cloud_buffer_.empty()){
      double dt = (now - cloud_buffer_.front().stamp).seconds();
      if(dt <= time_window) break;
      cloud_buffer_.pop_front();
    }
  }

  pcl::PointCloud<pcl::PointXYZ>::Ptr build_map()
  {
    pcl::PointCloud<pcl::PointXYZ>::Ptr merged(new pcl::PointCloud<pcl::PointXYZ>);
    for(const auto& c : cloud_buffer_){
      *merged += *(c.cloud_world);
    }
    return merged;
  }

  void build_grid_and_minz(const pcl::PointCloud<pcl::PointXYZ>& cloud,
                           std::map<std::pair<int,int>, Cell>& cells,
                           double bx, double by, double cell_size)
  {
    cells.clear();

    for(int i = 0; i < (int)cloud.points.size(); ++i){
      const auto& pt = cloud.points[i];
      if(!std::isfinite(pt.x) || !std::isfinite(pt.y) || !std::isfinite(pt.z)) continue;

      double xl = (double)pt.x - bx;
      double yl = (double)pt.y - by;

      int ix = (int)std::floor(xl / cell_size);
      int iy = (int)std::floor(yl / cell_size);

      Cell& c = cells[{ix, iy}];
      c.ix = ix;
      c.iy = iy;
      c.indices.push_back(i);

      if((double)pt.z < c.min_z) c.min_z = (double)pt.z;
      if((double)pt.z > c.z_max) c.z_max = (double)pt.z;
    }
  }

  void euclidean_cluster(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud,
                         std::vector<pcl::PointIndices>& cluster_indices)
  {
    cluster_indices.clear();

    const double tol = this->get_parameter("cluster_tolerance").as_double();
    const int min_sz = this->get_parameter("min_cluster_size").as_int();
    const int max_sz = this->get_parameter("max_cluster_size").as_int();

    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
    tree->setInputCloud(cloud);

    pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
    ec.setClusterTolerance(tol);
    ec.setMinClusterSize(min_sz);
    ec.setMaxClusterSize(max_sz);
    ec.setSearchMethod(tree);
    ec.setInputCloud(cloud);
    ec.extract(cluster_indices);
  }

  template<typename CloudPtrT>
  void finalize_cloud(CloudPtrT& cloud)
  {
    cloud->width = (uint32_t)cloud->points.size();
    cloud->height = 1;
    cloud->is_dense = true;
  }

  void publish_all_nonground(const sensor_msgs::msg::PointCloud2& in_msg,
                            const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud)
  {
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr colored(new pcl::PointCloud<pcl::PointXYZRGB>);
    colored->points.reserve(cloud->points.size());
    for(const auto& pt : cloud->points){
      pcl::PointXYZRGB p;
      p.x = pt.x; p.y = pt.y; p.z = pt.z;
      p.r = 255; p.g = 255; p.b = 255;
      colored->points.push_back(p);
    }
    finalize_cloud(colored);
    sensor_msgs::msg::PointCloud2 msg_col;
    pcl::toROSMsg(*colored, msg_col);
    msg_col.header = in_msg.header;
    pub_colored_->publish(msg_col);

    sensor_msgs::msg::PointCloud2 msg_ng;
    pcl::toROSMsg(*cloud, msg_ng);
    msg_ng.header = in_msg.header;
    pub_unknown_->publish(msg_ng);
  }

  double cluster_min_z(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cand_cloud,
                       const std::vector<pcl::PointIndices>& cluster_indices,
                       int k)
  {
    double mz = std::numeric_limits<double>::infinity();
    for(int cand_idx : cluster_indices[k].indices){
      const auto& p = cand_cloud->points[cand_idx];
      if(!std::isfinite(p.z)) continue;
      if((double)p.z < mz) mz = (double)p.z;
    }
    if(!std::isfinite(mz)){
      return std::numeric_limits<double>::quiet_NaN();
    }
    return mz;
  }

  bool detect_cylinder_in_roi(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud_map,
                              pcl::search::KdTree<pcl::PointXYZ>::Ptr& kdtree,
                              float xc, float yc, float roi_radius,
                              float zmin, float zmax,
                              int min_roi_pts, int normal_k,
                              float dist_thresh, float radius_min, float radius_max,
                              int min_inliers)
  {
    pcl::PointXYZ query;
    query.x = xc; query.y = yc; query.z = 0.0f;
    std::vector<int> idx;
    std::vector<float> dist2;
    if(kdtree->radiusSearch(query, roi_radius, idx, dist2) <= 0){
      return false;
    }

    pcl::PointCloud<pcl::PointXYZ>::Ptr roi(new pcl::PointCloud<pcl::PointXYZ>);
    roi->points.reserve(idx.size());

    for(int i : idx){
      const auto& p = cloud_map->points[i];
      if(!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z)) continue;
      if(p.z < zmin || p.z > zmax) continue;
      roi->points.push_back(p);
    }
    roi->width = (uint32_t)roi->points.size();
    roi->height = 1;
    roi->is_dense = true;
    if((int)roi->points.size() < min_roi_pts){
      return false;
    }

    pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
    ne.setInputCloud(roi);
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree_roi(new pcl::search::KdTree<pcl::PointXYZ>);
    ne.setSearchMethod(tree_roi);
    ne.setKSearch(normal_k);

    pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
    ne.compute(*normals);

    // NOTE:
    // PCL環境によっては SACSegmentationFromNormals が sac_segmentation.h 内にあります。
    // もし環境で見つからない場合は、PCLのインストール/ヘッダ構成を確認してください。
    pcl::SACSegmentationFromNormals<pcl::PointXYZ, pcl::Normal> seg;
    seg.setOptimizeCoefficients(true);
    seg.setModelType(pcl::SACMODEL_CYLINDER);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setNormalDistanceWeight(0.1);
    seg.setMaxIterations(200);
    seg.setDistanceThreshold(dist_thresh);
    seg.setRadiusLimits(radius_min, radius_max);
    seg.setInputCloud(roi);
    seg.setInputNormals(normals);

    pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
    pcl::ModelCoefficients::Ptr coeff(new pcl::ModelCoefficients);
    seg.segment(*inliers, *coeff);

    if(inliers->indices.empty()) return false;
    if((int)inliers->indices.size() < min_inliers) return false;
    return true;
  }

  void weed_callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
  {
    pcl::PointCloud<pcl::PointXYZ>::Ptr w(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::fromROSMsg(*msg, *w);
    // NaN除去（念のため）
    pcl::PointCloud<pcl::PointXYZ>::Ptr wf(new pcl::PointCloud<pcl::PointXYZ>);
    std::vector<int> nan;
    pcl::removeNaNFromPointCloud(*w, *wf, nan);

    {
      std::lock_guard<std::mutex> lk(weed_mtx_);
      *weed_cloud_ = *wf;
      has_weed_ = !weed_cloud_->points.empty();
      if(has_weed_){
        weed_kdtree_->setInputCloud(weed_cloud_);
      }
    }
    RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
                       "weed cloud updated: pts=%zu", wf->points.size());
  }

  // NOTE: ここは「地面平面」として c.n_ground/c.d_ground を使う（新方針の代表法線）
  bool lookup_ground_z_from_cells(double x, double y,
                                  const std::map<std::pair<int,int>, Cell>& cells,
                                  double bx, double by, double cell_size,
                                  double ref_R, double flat_ref_deg_max,
                                  float& gz_out) const
  {
    Eigen::Vector3f z_axis(0,0,1);

    int ix = (int)std::floor((x - bx) / cell_size);
    int iy = (int)std::floor((y - by) / cell_size);

    // 0) 自セルの代表地面平面
    auto it0 = cells.find({ix, iy});
    if(it0 != cells.end()){
      const Cell& c0 = it0->second;
      if(c0.has_ground_plane){
        return plane_z_at_xy(c0.n_ground, c0.d_ground, (float)x, (float)y, gz_out);
      }
    }

    const int ref_range = std::max(1, (int)std::ceil(ref_R / cell_size));

    auto search_nearest_ground_plane =
      [&](bool require_flat, Eigen::Vector3f& n_ref, float& d_ref) -> bool
    {
      bool found = false;
      float best_dist = std::numeric_limits<float>::infinity();

      for(int dy=-ref_range; dy<=ref_range; ++dy){
        for(int dx=-ref_range; dx<=ref_range; ++dx){
          auto it = cells.find({ix + dx, iy + dy});
          if(it == cells.end()) continue;
          const Cell& nb = it->second;
          if(!nb.has_ground_plane) continue;

          double dist = std::sqrt((double)(dx*dx + dy*dy)) * cell_size;
          if(dist > ref_R) continue;

          if(require_flat){
            double tilt = angle_deg(nb.n_ground, z_axis);
            if(tilt > flat_ref_deg_max) continue;
          }

          if((float)dist < best_dist){
            best_dist = (float)dist;
            n_ref = nb.n_ground;
            d_ref = nb.d_ground;
            found = true;
          }
        }
      }
      return found;
    };

    // 1) 最寄り（斜面OK）
    {
      Eigen::Vector3f n_ref = z_axis;
      float d_ref = 0.0f;
      if(search_nearest_ground_plane(false, n_ref, d_ref)){
        return plane_z_at_xy(n_ref, d_ref, (float)x, (float)y, gz_out);
      }
    }

    // 2) flatのみ（保険）
    {
      Eigen::Vector3f n_ref = z_axis;
      float d_ref = 0.0f;
      if(search_nearest_ground_plane(true, n_ref, d_ref)){
        return plane_z_at_xy(n_ref, d_ref, (float)x, (float)y, gz_out);
      }
    }
    return false;
  }

  pcl::PointCloud<pcl::PointXYZ>::Ptr roi_filter_in_base(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud_world,
    const geometry_msgs::msg::TransformStamped& tf_w_b,
    float x_min, float x_max, float y_min, float y_abs)
  {
    pcl::PointCloud<pcl::PointXYZ>::Ptr out(new pcl::PointCloud<pcl::PointXYZ>);
    out->points.reserve(cloud_world->points.size());

    tf2::Quaternion q(tf_w_b.transform.rotation.x, tf_w_b.transform.rotation.y,
                      tf_w_b.transform.rotation.z, tf_w_b.transform.rotation.w);
    tf2::Matrix3x3 R(q);

    const float bx = (float)tf_w_b.transform.translation.x;
    const float by = (float)tf_w_b.transform.translation.y;
    const float bz = (float)tf_w_b.transform.translation.z;

    for(const auto& p : cloud_world->points){
      if(!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z)) continue;

      // world -> base : pb = R^T * (pw - b)
      tf2::Vector3 pw(p.x - bx, p.y - by, p.z - bz);
      tf2::Vector3 pb = R.transpose() * pw;

      const float xb = (float)pb.x();
      const float yb = (float)pb.y();

      if(xb < x_min) continue;
      if(xb > x_max) continue;
      if(yb < y_min) continue;
      if(yb > y_abs) continue;

      out->points.push_back(p);
    }

    out->width = (uint32_t)out->points.size();
    out->height = 1;
    out->is_dense = true;
    return out;
  }

private:
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_weed_;

  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_colored_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_ground_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_unknown_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_cylinder_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_cell_ground_;

  tf2_ros::Buffer tf_buffer_;
  tf2_ros::TransformListener tf_listener_;

  std::deque<TimeCloud> cloud_buffer_;

  pcl::PointCloud<pcl::PointXYZ>::Ptr weed_cloud_{new pcl::PointCloud<pcl::PointXYZ>};
  pcl::search::KdTree<pcl::PointXYZ>::Ptr weed_kdtree_{new pcl::search::KdTree<pcl::PointXYZ>};
  bool has_weed_ = false;
  std::mutex weed_mtx_;
};

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<IdentificationNode>());
  rclcpp::shutdown();
  return 0;
}
