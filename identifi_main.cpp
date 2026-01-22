#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>

#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/filter.h>

#include <pcl/segmentation/sac_segmentation.h>

#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>

#include <Eigen/Dense>

#include <deque>
#include <map>
#include <vector>
#include <limits>
#include <algorithm>
#include <cmath>

struct Cell
{
    int ix = 0;
    int iy = 0;
    std::vector<int> indices;
    std::vector<float> zs;
    
    double z_seed = std::numeric_limits<double>::quiet_NaN();
    Eigen::Vector3f n = Eigen::Vector3f(0.0f, 0.0f, 1.0f);
    float d = 0.0f;
    bool plane_on = false;
    double tilt_deg = std::numeric_limits<double>::quiet_NaN();
};

struct TimeCloud
{
  rclcpp::Time stamp;
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_world;
};

double deg2rad(double deg)
{
    return deg * 3.14159265358979323846 / 180.0;
}

double rad2deg(double rad)
{
    return rad * 180.0 / 3.14159265358979323846;
}

double percentile_simple(std::vector<float> v, double per)
{
    if(v.empty()) return std::numeric_limits<double>::quiet_NaN();

    std::sort(v.begin(), v.end());
    int n = (int)v.size();
    int k = (int)std::floor(per * (double)(n - 1));
    return (double)v[k];
}

bool plane_z_at_xy_simple(const Eigen::Vector3f& n, float d, double x, double y, double& z_out)
{
    double nx = (double)n.x();
    double ny = (double)n.y();
    double nz = (double)n.z();

    if(!std::isfinite(nx) || !std::isfinite(ny) || !std::isfinite(nz)) return false;
    if(std::fabs(nz) < 1e-6) return false;

    z_out = (-(nx * x + ny * y + (double)d) / nz);
    return std::isfinite(z_out);
}

double plane_tilt_deg_from_normal_simple(Eigen::Vector3f n)
{
    if(n.norm() < 1e-6f) return std::numeric_limits<double>::quiet_NaN();
    n.normalize();

    double nz = (double)n.z();
    return rad2deg(std::acos(nz));
}

class IdentificationNode : public rclcpp::Node
{
public:
  IdentificationNode()
  : Node("Identification_node"),tf_buffer_(this->get_clock()),tf_listener_(tf_buffer_)
  {
    sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
      "/cloud_registered", rclcpp::SensorDataQoS(),
      std::bind(&IdentificationNode::cloud_callback, this, std::placeholders::_1));
    
    pub_colored_  = this->create_publisher<sensor_msgs::msg::PointCloud2>("/Identification", 10);
    pub_ground_   = this->create_publisher<sensor_msgs::msg::PointCloud2>("/Identification_ground", 10);
    pub_slope_    = this->create_publisher<sensor_msgs::msg::PointCloud2>("/Identification_slope", 10);
    pub_step_     = this->create_publisher<sensor_msgs::msg::PointCloud2>("/Identification_step", 10);
    pub_obstacle_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/Identification_obstacle", 10);
    pub_unknown_  = this->create_publisher<sensor_msgs::msg::PointCloud2>("/Identification_unknown", 10);

    this->declare_parameter("world_frame", "camera_init");
    this->declare_parameter("base_frame", "body");

    this->declare_parameter("voxel_size", 0.05);
    this->declare_parameter("time_window", 5.0);

    this->declare_parameter("cell_size", 0.20);
    this->declare_parameter("min_points_per_cell", 10);

    this->declare_parameter("patch_range", 1);
    this->declare_parameter("ground_percentile", 0.20);
    this->declare_parameter("seed_margin", 0.05);

    this->declare_parameter("ransac_dist_thr", 0.05);
    this->declare_parameter("ransac_max_iter", 200);
    this->declare_parameter("ground_plane_max_deg", 40.0);

    this->declare_parameter("angle_fla_slo", 10.0);
    this->declare_parameter("slope_max_deg", 30.0);

    this->declare_parameter("step_thr", 0.05);
    this->declare_parameter("h_obs", 0.15);

    this->declare_parameter("ground_h_thr", 0.05);

    RCLCPP_INFO(this->get_logger(), "Readable patch-ground segmentation node initialized.");

  }

private:
  void cloud_callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
  {
    const std::string world_frame = this->get_parameter("world_frame").as_string();
    const std::string base_frame  = this->get_parameter("base_frame").as_string();

    // 座標情報を持った点群データをinputに格納
    pcl::PointCloud<pcl::PointXYZ>::Ptr input(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::fromROSMsg(*msg, *input);

    // 外れ点除去
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZ>);
    std::vector<int> nan_indices;
    pcl::removeNaNFromPointCloud(*input, *cloud_filtered, nan_indices);

    // tfでbaseframeをworldframeで取得
    const rclcpp::Time stamp(msg->header.stamp);
    geometry_msgs::msg::TransformStamped tf_w_b;
    try{
      tf_w_b = tf_buffer_.lookupTransform(world_frame, base_frame, stamp);
    }catch(const std::exception &e){
      RCLCPP_WARN(this->get_logger(), "TF lookup failed (%s -> %s): %s", base_frame.c_str(), world_frame.c_str(), e.what());
      return;
    }

    const double bx = tf_w_b.transform.translation.x;
    const double by = tf_w_b.transform.translation.y;

    // ボクセルダウンサンプリング
    const double voxel_size = this->get_parameter("voxel_size").as_double();
    pcl::PointCloud<pcl::PointXYZ>::Ptr down = voxel_downsample(cloud_filtered, voxel_size);

    //過去に取得した点群に新しい点群を積み上げる
    TimeCloud tc;
    tc.stamp = stamp;
    tc.cloud_world = down;
    cloud_buffer_.push_back(tc);

    //time_window秒以上前の点群は捨てる
    const double time_window = this->get_parameter("time_window").as_double();
    pop_old_clouds(stamp, time_window);

    //積み上げた点群を更にダウンサンプリング
    pcl::PointCloud<pcl::PointXYZ>::Ptr merged = build_map();
    pcl::PointCloud<pcl::PointXYZ>::Ptr ds_map = voxel_downsample(merged, voxel_size);

    const double cell_size = this->get_parameter("cell_size").as_double();
    const int min_pts_cell = this->get_parameter("min_points_per_cell").as_int();

    //グリッドを生成し、各グリッドに点の番号を格納
    std::map<std::pair<int,int>, Cell> cells;
    build_grid(*ds_map, cells, bx, by, cell_size);

    // --- seed height per cell ---
    const double seed_per = this->get_parameter("ground_percentile").as_double();
    for(auto& kv : cells){
      Cell& c = kv.second;
      if((int)c.indices.size() < min_pts_cell){
        c.plane_on = false;
        continue;
      }
      c.z_seed = percentile_simple(c.zs, seed_per);
    }

    // --- plane per cell ---
    const int patch_range = this->get_parameter("patch_range").as_int();
    const double seed_margin = this->get_parameter("seed_margin").as_double();
    const double ransac_dist_thr = this->get_parameter("ransac_dist_thr").as_double();
    const int ransac_max_iter = this->get_parameter("ransac_max_iter").as_int();
    const double ground_plane_max_deg = this->get_parameter("ground_plane_max_deg").as_double();

    for(auto& kv : cells){
      Cell& c = kv.second;

      if(!std::isfinite(c.z_seed)){
        c.plane_on = false;
        continue;
      }

      std::vector<int> cand;
      collect_patch_candidates(*ds_map, cells, c, patch_range, seed_margin, cand);

      if((int)cand.size() < 30){
        c.plane_on = false;
        continue;
      }

      Eigen::Vector3f n;
      float d;
      int inliers_n = 0;

      if(!fit_plane_ransac_indices_easy(*ds_map, cand, ransac_dist_thr, ransac_max_iter, n, d, inliers_n)){
        c.plane_on = false;
        continue;
      }

      // normalize and nz>=0
      if(n.norm() < 1e-6f){
        c.plane_on = false;
        continue;
      }
      n.normalize();
      if(n.z() < 0.0f){
        n = -n;
        d = -d;
      }

      double tilt = plane_tilt_deg_from_normal_simple(n);
      if(!std::isfinite(tilt) || tilt > ground_plane_max_deg){
        c.plane_on = false;
        continue;
      }

      c.n = n;
      c.d = d;
      c.plane_on = true;
      c.tilt_deg = tilt;
    }

    publish_result(*msg, ds_map, cells, bx, by, cell_size);
  }

  pcl::PointCloud<pcl::PointXYZ>::Ptr voxel_downsample(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr& input, double vx)
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

  void build_grid(const pcl::PointCloud<pcl::PointXYZ>& cloud, std::map<std::pair<int,int>, Cell>& cells, double bx, double by, double cell_size)
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
      c.zs.push_back((float)pt.z);
    }
  }

  void collect_patch_candidates(const pcl::PointCloud<pcl::PointXYZ>& cloud, const std::map<std::pair<int,int>, Cell>& cells, const Cell& center, int range, double seed_margin, std::vector<int>& out_indices)
  {
    out_indices.clear();

    for(int dy = -range; dy <= range; ++dy){
      for(int dx = -range; dx <= range; ++dx){
        auto it = cells.find({center.ix + dx, center.iy + dy});
        if(it == cells.end()) continue;

        const Cell& nb = it->second;
        if(!std::isfinite(nb.z_seed)) continue;

        double z_ref = nb.z_seed;

        for(int idx : nb.indices){
          const auto& pt = cloud.points[idx];
          if(!std::isfinite(pt.z)) continue;

          if((double)pt.z <= z_ref + seed_margin){
            out_indices.push_back(idx);
          }
        }
      }
    }

    std::sort(out_indices.begin(), out_indices.end());
    out_indices.erase(std::unique(out_indices.begin(), out_indices.end()), out_indices.end());
  }

  bool fit_plane_ransac_indices(const pcl::PointCloud<pcl::PointXYZ>& cloud, const std::vector<int>& indices, double dist_thr, int max_iter, Eigen::Vector3f& n_out, float& d_out, int& inliers_n_out)
  {
    inliers_n_out = 0;
    if((int)indices.size() < 30) return false;

    pcl::SACSegmentation<pcl::PointXYZ> seg;
    seg.setOptimizeCoefficients(true);
    seg.setModelType(pcl::SACMODEL_PLANE);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setDistanceThreshold(dist_thr);
    seg.setMaxIterations(max_iter);

    pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
    pcl::ModelCoefficients::Ptr coeff(new pcl::ModelCoefficients);

    pcl::IndicesPtr idx_ptr(new std::vector<int>(indices));
    seg.setInputCloud(cloud.makeShared());
    seg.setIndices(idx_ptr);

    seg.segment(*inliers, *coeff);
    if(inliers->indices.empty()) return false;
    if(coeff->values.size() < 4) return false;

    // 平面の式: ax + by + cz + d = 0
    float a = coeff->values[0];
    float b = coeff->values[1];
    float c = coeff->values[2];
    float d = coeff->values[3];

    Eigen::Vector3f n(a,b,c);
    if(n.norm() < 1e-6f) return false;

    n_out = n;
    d_out = d;
    inliers_n_out = (int)inliers->indices.size();
    return true;
  }

  void publish_result(const sensor_msgs::msg::PointCloud2& in_msg, const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, const std::map<std::pair<int,int>, Cell>& cells)
  {

    const double flat_max_deg  = this->get_parameter("angle_fla_slo").as_double();
    const double slope_max_deg = this->get_parameter("slope_max_deg").as_double();

    const double ground_h_thr = this->get_parameter("ground_h_thr").as_double();
    const double step_thr     = this->get_parameter("step_thr").as_double();
    const double h_obs        = this->get_parameter("h_obs").as_double();

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr colored(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr ground(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr slope(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr step(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr obstacle(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr unknown(new pcl::PointCloud<pcl::PointXYZ>);

    colored->points.reserve(cloud->points.size());

    for(const auto& kv : cells){
      const Cell& c = kv.second;

      for(int idx : c.indices){
        const auto& pt = cloud->points[idx];

        //デフォルト：unknown(オレンジ)
        uint8_t r=255, g=140, b=0;
        bool is_unknown = true;

        if(c.plane_on){
          //z_pred：地面高さ
          double z_pred;
          if(plane_z_at_xy(c.n, c.d, (double)pt.x, (double)pt.y, z_pred)){
            double h = (double)pt.z - z_pred;

            is_unknown = false;

            //斜面と見なす角度
            bool is_slope = (c.tilt_deg > flat_max_deg);
            //障害物と見なす角度
            bool too_steep = (c.tilt_deg > slope_max_deg);

            if(too_steep){
              // if "support plane" is steep, treat everything as obstacle (safety)
              r=255; g=0; b=0;
              obstacle->points.push_back(pt);
            }
            else{
              if(h >= h_obs){
                r=255; g=0; b=0;   // obstacle
                obstacle->points.push_back(pt);
              }
              else if(h >= step_thr){
                r=255; g=0; b=255; // step (traversable)
                step->points.push_back(pt);
              }
              else if(std::fabs(h) <= ground_h_thr){
                if(is_slope){
                  r=0; g=0; b=255; // slope
                  slope->points.push_back(pt);
                }else{
                  r=0; g=255; b=0; // flat ground
                  ground->points.push_back(pt);
                }
              }
              else{
                // slight mismatch points (e.g., vegetation low) -> mark as step-ish or unknown?
                // Here: treat as step-ish if above 0, else ground.
                if(h > 0.0){
                  r=255; g=0; b=255;
                  step->points.push_back(pt);
                }else{
                  r=0; g=255; b=0;
                  ground->points.push_back(pt);
                }
              }
            }
          }
        }

        if(is_unknown){
          unknown->points.push_back(pt);
        }

        pcl::PointXYZRGB prgb;
        prgb.x = pt.x; prgb.y = pt.y; prgb.z = pt.z;
        prgb.r = r; prgb.g = g; prgb.b = b;
        colored->points.push_back(prgb);
      }
    }

    colored->width = (uint32_t)colored->points.size();
    colored->height = 1;
    ground->width = (uint32_t)ground->points.size(); ground->height = 1;
    slope->width = (uint32_t)slope->points.size(); slope->height = 1;
    step->width  = (uint32_t)step->points.size();  step->height = 1;
    obstacle->width = (uint32_t)obstacle->points.size(); obstacle->height = 1;
    unknown->width = (uint32_t)unknown->points.size(); unknown->height = 1;

    sensor_msgs::msg::PointCloud2 msg_col, msg_g, msg_s, msg_step, msg_obs, msg_unk;
    pcl::toROSMsg(*colored, msg_col);
    pcl::toROSMsg(*ground,  msg_g);
    pcl::toROSMsg(*slope,   msg_s);
    pcl::toROSMsg(*step,    msg_step);
    pcl::toROSMsg(*obstacle,msg_obs);
    pcl::toROSMsg(*unknown, msg_unk);

    msg_col.header = in_msg.header;
    msg_g.header   = in_msg.header;
    msg_s.header   = in_msg.header;
    msg_step.header= in_msg.header;
    msg_obs.header = in_msg.header;
    msg_unk.header = in_msg.header;

    pub_colored_->publish(msg_col);
    pub_ground_->publish(msg_g);
    pub_slope_->publish(msg_s);
    pub_step_->publish(msg_step);
    pub_obstacle_->publish(msg_obs);
    pub_unknown_->publish(msg_unk);
  }

private:
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_;

  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_colored_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_ground_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_slope_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_step_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_obstacle_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_unknown_;

  tf2_ros::Buffer tf_buffer_;
  tf2_ros::TransformListener tf_listener_;

  std::deque<TimeCloud> cloud_buffer_;
};

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<IdentificationNode>());
  rclcpp::shutdown();
  return 0;
}
