#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl/features/normal_3d.h>
#include <cmath>
#include <rclcpp/qos.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <limits>
#include <algorithm>
#include <deque>
#include <pcl/filters/crop_box.h>
#include <pcl/filters/filter.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.hpp>

struct Cell
{
    int ix = 0;
    int iy = 0;
    int count = 0;
    double z_max = -std::numeric_limits<double>::infinity();
    double z_min = std::numeric_limits<double>::infinity();
    double z_main;

    bool obstacle = false;
    bool slope = false;
    bool unknown = false;
    bool around_unknowm = false;

    double z_gro_per = std::numeric_limits<double>::quiet_NaN();
    double z_rou_per = std::numeric_limits<double>::quiet_NaN();
    double ground_z = std::numeric_limits<double>::quiet_NaN();
    double roughness = std::numeric_limits<double>::quiet_NaN();

    std::vector<float> points_z;

    std::vector<int> pc_indices;

    bool ground_candidate = false;
    double ground_confidence = 0.0;
    bool detern_ground = false;
};

struct TimeCloud
{
  rclcpp::Time stamp;
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_world;
};

class IdentificationNode : public rclcpp::Node
{
public:
  IdentificationNode()
  : Node("Identification_node"),tf_buffer_(this->get_clock()),tf_listener_(tf_buffer_)
  {
    sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
      "/cloud_registered", rclcpp::SensorDataQoS(),
      std::bind(&IdentificationNode::cloud_callback, this, std::placeholders::_1));

    pub_colored = this->create_publisher<sensor_msgs::msg::PointCloud2>("/Identification", 10);
    pub_obstacle = this->create_publisher<sensor_msgs::msg::PointCloud2>("/Identification_obstacle", 10);
    pub_slope = this->create_publisher<sensor_msgs::msg::PointCloud2>("/Identification_slope", 10);
    pub_around_unknown = this->create_publisher<sensor_msgs::msg::PointCloud2>("/Identification_around_unknown", 10);
    pub_ground = this->create_publisher<sensor_msgs::msg::PointCloud2>("/Identification_ground", 10);
    pub_undergrowth = this->create_publisher<sensor_msgs::msg::PointCloud2>("/Identification_undergrowth", 10);

    this->declare_parameter("world_frame", "camera_init");
    this->declare_parameter("base_frame", "body");

    //ダウンサンプリングの際のボクセルサイズ
    this->declare_parameter("voxel_size", 0.05);
    // cellの1辺の長さ（m）
    this->declare_parameter("cell_size", 0.20);
    //点群を保持する秒数[s]
    this->declare_parameter("time_window",5.0);

    // 高さのしきい値（障害物・坂道の判定で使用）
    this->declare_parameter("height_threshold", 0.1);

    // 角度のしきい値（平地と坂道の境界となる角度）
    this->declare_parameter("angle_fla_slo", 10.0);

    this->declare_parameter("ground_percentile", 0.10);
    this->declare_parameter("rough_percentile", 0.90);
    this->declare_parameter("min_points_per_cell", 10);
    //想定される1セル内における点の数の最大
    this->declare_parameter("max_points_per_cell", 20);

    //ground候補と見なす粗さの許容値上限
    this->declare_parameter("rough_ground_max", 0.12);
    //BTSで地面領域を広げる際のconfの許容値
    this->declare_parameter("conf_min", 0.6);
    //連続性をチェック
    this->declare_parameter("step_thr", 0.08);
    this->declare_parameter("theta_max", 25.0);
    //点の高さで草・障害物の分類
    this->declare_parameter("h_undergrowth", 0.05);
    this->declare_parameter("h_obs", 0.20);

    // 周辺のセル何マスと比較するか指定（1なら周囲8マス、2なら周囲24マス）
    this->declare_parameter("check_range", 1);
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
    }
    catch(const std::exception &e){
      RCLCPP_WARN(this->get_logger(),"TF lookup failed (%s -> %s): %s",base_frame.c_str(), world_frame.c_str(), e.what());
      return;
    }
    const double bx = tf_w_b.transform.translation.x;
    const double by = tf_w_b.transform.translation.y;

    // ボクセルダウンサンプリング
    const double voxel_size = this->get_parameter("voxel_size").as_double();
    pcl::PointCloud<pcl::PointXYZ>::Ptr downsampled_cloud = voxel_downsample(cloud_filtered,voxel_size);

    //過去に取得した点群に新しい点群を積み上げる
    TimeCloud tc;
    tc.stamp = stamp;
    tc.cloud_world = downsampled_cloud;
    cloud_buffer_.push_back(tc);

    //time_window秒以上前の点群は捨てる
    const double time_window = this ->get_parameter("time_window").as_double();
    pop_old_clouds(stamp,time_window);

    //積み上げた点群を更にダウンサンプリング
    pcl::PointCloud<pcl::PointXYZ>::Ptr map_cloud = build_map();
    pcl::PointCloud<pcl::PointXYZ>::Ptr ds_map = voxel_downsample(map_cloud, voxel_size);

    const double cell_size = this->get_parameter("cell_size").as_double();
    std::map<std::pair<int, int>, Cell> cell_info;

    //グリッドを生成し、各グリッドに点の番号を格納
    build_grid(*ds_map, cell_info, bx, by, cell_size);

    //周囲のグリッドの情報と比較し、障害物や斜面の識別を行う
    detect_obstacle_slope(cell_info, cell_size);

    //点群を出力するための関数
    publish_result(*msg, ds_map, cell_info);
  }

  pcl::PointCloud<pcl::PointXYZ>::Ptr voxel_downsample(const pcl::PointCloud<pcl::PointXYZ>::Ptr& input, double vx_size)
  {
    pcl::VoxelGrid<pcl::PointXYZ> voxel;
    voxel.setInputCloud(input);
    voxel.setLeafSize(static_cast<float>(vx_size),static_cast<float>(vx_size),static_cast<float>(vx_size));
    pcl::PointCloud<pcl::PointXYZ>::Ptr downsampled_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    voxel.filter(*downsampled_cloud);
    return downsampled_cloud;
  }

  void pop_old_clouds(const rclcpp::Time& now, double time_window)
  {
    while(!cloud_buffer_.empty()){
      const double dif_time = (now - cloud_buffer_.front().stamp).seconds();
      if(dif_time <= time_window){
        break;
      }
      cloud_buffer_.pop_front();
    }
  }

  pcl::PointCloud<pcl::PointXYZ>::Ptr build_map()
  {
    pcl::PointCloud<pcl::PointXYZ>::Ptr merged(new pcl::PointCloud<pcl::PointXYZ>);
    merged->points.reserve(200000);
    for(auto& c : cloud_buffer_){
     *merged += *(c.cloud_world);
    }
    return merged;
  }

  void build_grid(const pcl::PointCloud<pcl::PointXYZ>& cloud_world, std::map<std::pair<int, int>, Cell>& cells, double bx, double by, double cell_size)
  {
    cells.clear();
    int max_count = 0;
    for(int i = 0; i < static_cast<int>(cloud_world.points.size()); ++i){
      const auto& ptw =cloud_world.points[i];
      const float xl = static_cast<float>(ptw.x - bx);
      const float yl = static_cast<float>(ptw.y - by);
      const float zl = static_cast<float>(ptw.z);

      const int ix = static_cast<int>(std::floor(xl / cell_size));
      const int iy = static_cast<int>(std::floor(yl / cell_size));
      const std::pair<int,int> key{ix, iy};

      auto& cell = cells[key];
      cell.ix = ix;
      cell.iy = iy;
      if(cell.z_max < zl){
        cell.z_max = zl;
      }
      if(cell.z_min > zl){
        cell.z_min = zl;
        cell.z_main = cell.z_min;
      }
      cell.count ++;
      cell.pc_indices.push_back(i);
      cell.points_z.push_back(zl);
      if(cell.count > max_count){
        max_count = cell.count;
      }
    }
    RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 1000,"max_count=%d", max_count);    
  }

  double percentile(std::vector<float> samples, double per_num)
  {
    if(samples.size() == 0){
      return std::numeric_limits<double>::quiet_NaN();
    }
    int n = static_cast<int>(samples.size());
    double index_per_num = per_num * static_cast<double>(n - 1);
    int k = static_cast<int>(std::floor(index_per_num));
    //部分手なパーセンタイル
    std::nth_element(samples.begin(), samples.begin() + k, samples.end());
    double result = static_cast<double>(samples[k]);
    return result;
  }

  void detect_obstacle_slope(std::map<std::pair<int, int>, Cell>& cells, double cell_size)
  {
    const int min_pts = this->get_parameter("min_points_per_cell").as_int();
    const int max_pts = this->get_parameter("max_points_per_cell").as_int();
    const double per_gro = this->get_parameter("ground_percentile").as_double();
    const double per_rou = this->get_parameter("rough_percentile").as_double();
    const double rough_ground_max = this->get_parameter("rough_ground_max").as_double();
    const double conf_min = this->get_parameter("conf_min").as_double();
    //const double step_thr = this->get_parameter("step_thr").as_double();
    const double theta_max = this->get_parameter("theta_max").as_double();
    const double angle_fla_slo = this->get_parameter("angle_fla_slo").as_double();
    const int check_range = this->get_parameter("check_range").as_int();

    double max_ratio = static_cast<double>(max_pts) / static_cast<double>(min_pts); 
    std::deque<std::pair<int,int>> q;
    const double PI = 3.14159265358979323846;

    for (auto& kv : cells){
      auto& c = kv.second;
      if (c.count < min_pts){
        c.unknown = true;
        continue;
      }
      c.z_gro_per = percentile(c.points_z, per_gro);
      c.z_rou_per = percentile(c.points_z, per_rou);
      if(!std::isfinite(c.z_gro_per) || !std::isfinite(c.z_rou_per)){
        c.unknown = true;
        continue;
      }
      c.roughness = c.z_rou_per - c.z_gro_per;

      if(c.roughness < rough_ground_max){
        double ratio_pts = static_cast<double>(c.count) / static_cast<double>(min_pts);
        c.ground_candidate = true;
        double pts_score = ratio_pts / max_ratio;
        double rou_score = 1.0 - (c.roughness / rough_ground_max);
        c.ground_confidence = pts_score * rou_score;
        if(c.ground_confidence >= conf_min){
          c.detern_ground = true;
          q.push_back(kv.first);
        }
      }
    }

    const int dx[4] = {1, -1, 0, 0};
    const int dy[4] = {0, 0, 1, -1};
    while(!q.empty()){
      auto key = q.front();
      q.pop_front();
      Cell& base_info = cells.find(key)->second;
      double z_up_base = base_info.z_gro_per;

      for(int k=0;k<4;k++){
        std::pair<int, int> next_key{key.first + dx[k], key.second + dy[k]};
        if(cells.find(next_key) == cells.end()){
          continue;
        }
        Cell& next_info = cells.find(next_key)->second;
        if(next_info.detern_ground){
          continue;
        }
        if(!next_info.ground_candidate){
          continue;
        }
        if(!std::isfinite(next_info.z_gro_per)){
          continue;
        }
        double z_up_next = next_info.z_gro_per;
        double allow_h = std::tan(theta_max * PI / 180) * cell_size;
        if(std::abs(z_up_base - z_up_next) <= allow_h){
          next_info.detern_ground = true;
          q.push_back(next_key);
        }
      }
    }

    for (auto& kv : cells){
      auto& c = kv.second;
      if(c.detern_ground){
        c.ground_z = c.z_gro_per;
        continue;
      }
      double best = std::numeric_limits<double>::infinity();
      double best_z = std::numeric_limits<double>::quiet_NaN();
      for(int dy = -check_range; dy <= check_range; dy++){
        for(int dx = -check_range; dx <= check_range; dx++){
          if(dx == 0 && dy == 0){
            continue;
          }
          auto it = cells.find({c.ix + dx, c.iy + dy}) ;
          if(it == cells.end()){
            continue;
          }
          Cell& next_info = it->second;
          if(!std::isfinite(next_info.ground_z)){
            continue;
          }
          double distance = (double)dx*dx + (double)dy*dy;
          if(distance < best){
            best = distance;
            best_z = next_info.ground_z;
          }
        }
      }
      if(std::isfinite(best_z)){
        c.ground_z = best_z;
      }
      else{
        c.unknown = true;
      }
    }

    for (auto& kv : cells){
      Cell& c = kv.second;
      if(c.detern_ground == false){
        continue;
      }
      bool xplus_cell = false;
      bool xminus_cell = false;
      bool yplus_cell = false;
      bool yminus_cell = false;
      double xplus_z, xminus_z, yplus_z, yminus_z;

      for(int k = 0; k < 4; k++){
        std::pair<int, int> compare_key{c.ix + dx[k], c.iy + dy[k]};
        if(cells.find(compare_key) == cells.end()){
          continue;
        }
        Cell& next_cell = cells.find(compare_key)->second;
        if(next_cell.detern_ground){
          continue;
        }
        if(!std::isfinite(next_cell.ground_z)){
          continue;
        }
        if(k == 0){
          xplus_cell = true;
          xplus_z = next_cell.ground_z;
        }
        if(k == 1){
          xminus_cell = true;
          xminus_z = next_cell.ground_z;
        }
        if(k == 2){
          yplus_cell = true;
          yplus_z = next_cell.ground_z;
        }
        if(k == 3){
          yminus_cell = true;
          yminus_z = next_cell.ground_z;
        }
      }
      double dzdx = 0.0;
      double dzdy = 0.0;
      if(xplus_cell && xminus_cell){
        dzdx = (xplus_z - xminus_z) / (2.0 * cell_size);
      }
      if(yplus_cell && yminus_cell){
        dzdy = (yplus_z - yminus_z) / (2.0 * cell_size);
      }
      double angle = std::atan(std::sqrt(dzdx * dzdx + dzdy * dzdy)) * (180.0 / PI);
      if(angle > angle_fla_slo){
        c.slope = true;
      }
    }       
  }

  void publish_result(const sensor_msgs::msg::PointCloud2& in_msg, const pcl::PointCloud<pcl::PointXYZ>::Ptr& d_map, const std::map<std::pair<int, int>, Cell>& cells){
    const double h_undergrowth = this->get_parameter("h_undergrowth").as_double();
    const double h_obs = this->get_parameter("h_obs").as_double();
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr colored_pc(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr obstacle(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr slope(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr around_unknown(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr ground(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr undergrowth(new pcl::PointCloud<pcl::PointXYZ>);
    colored_pc->points.reserve(d_map->points.size());
    obstacle->points.reserve(d_map->points.size() / 2);
    slope->points.reserve(d_map->points.size() / 2);
    for(auto& kv : cells){
      auto& c = kv.second;
      for(auto& indx : c.pc_indices){
        auto& pt = d_map->points[indx];
        uint8_t r = 0;
        uint8_t g = 255;
        uint8_t b = 0;
        if(c.unknown || !std::isfinite(c.ground_z)){
          r = 255;
          g = 140;
          b = 0;
          around_unknown->points.push_back(pt); 
        }
        else{
          double h_diff = (double)pt.z - c.ground_z;
          if(h_diff > h_obs){
            r = 255;
            g = 0;
            b = 0;
            obstacle->points.push_back(pt);
          }
          else if(h_diff > h_undergrowth){
            //yellow
            r = 255;
            g = 255;
            b = 0;
            undergrowth->points.push_back(pt);
          }
          else if(c.slope){
            r = 0;
            g = 0;
            b = 255;
            slope->points.push_back(pt);
          }
          else{
            ground->points.push_back(pt);
          }
        }
        pcl::PointXYZRGB p_info;
        p_info.x = pt.x;
        p_info.y = pt.y;
        p_info.z = pt.z;
        p_info.r = r;
        p_info.g = g;
        p_info.b = b;
        colored_pc->points.push_back(p_info);
      }      
    }
    colored_pc-> width = colored_pc -> points.size();
    colored_pc -> height = 1;
    obstacle -> width = obstacle -> points.size();
    obstacle -> height = 1;
    slope -> width = slope -> points.size();
    slope -> height = 1;
    around_unknown -> width = around_unknown -> points.size();
    around_unknown -> height = 1;
    undergrowth -> width = undergrowth -> points.size();
    undergrowth -> height = 1;
    ground -> width = ground -> points.size();
    ground -> height = 1;
    
    sensor_msgs::msg::PointCloud2 col_msg, obs_msg, slo_msg, around_unknown_msg, ground_msg, undergrowth_msg;
    pcl::toROSMsg(*colored_pc, col_msg);
    pcl::toROSMsg(*obstacle, obs_msg);
    pcl::toROSMsg(*slope, slo_msg);
    pcl::toROSMsg(*around_unknown, around_unknown_msg);
    pcl::toROSMsg(*ground, ground_msg);
    pcl::toROSMsg(*undergrowth, undergrowth_msg);
    col_msg.header = in_msg.header;  
    obs_msg.header = in_msg.header;
    slo_msg.header = in_msg.header; 
    around_unknown_msg.header = in_msg.header; 
    ground_msg.header = in_msg.header;
    undergrowth_msg.header = in_msg.header;
    pub_colored -> publish(col_msg);
    pub_obstacle -> publish(obs_msg);
    pub_slope -> publish(slo_msg);
    pub_around_unknown -> publish(around_unknown_msg);
    pub_ground->publish(ground_msg);
    pub_undergrowth->publish(undergrowth_msg);
  }

private:
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_colored;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_obstacle;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_slope;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_around_unknown;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_ground;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_undergrowth;

  tf2_ros::Buffer tf_buffer_;
  tf2_ros::TransformListener tf_listener_;

  std::deque<TimeCloud> cloud_buffer_;
};

int main(int argc, char **argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<IdentificationNode>());
  rclcpp::shutdown();
  return 0;
}
