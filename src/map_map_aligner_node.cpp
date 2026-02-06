// ============================
// [B MODE + POINT TRANSFORM PUBLISH] map_map_aligner_node.cpp
// ============================

#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <geometry_msgs/TransformStamped.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_ros/static_transform_broadcaster.h>

#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/registration/ndt.h>
#include <pcl/common/transforms.h>   // [ADD] transformPointCloud

#include <Eigen/Dense>
#include <limits>
#include <cmath>
#include <stdexcept>

using PointT = pcl::PointXYZ;
using CloudT = pcl::PointCloud<PointT>;

class MapMapAligner
{
public:
  MapMapAligner(ros::NodeHandle& nh, ros::NodeHandle& pnh)
  : nh_(nh), pnh_(pnh)
  {
    pnh_.param<std::string>("map1_pcd", map1_pcd_, std::string(""));
    pnh_.param<std::string>("map2_topic", map2_topic_, std::string("/map2"));

    // frame_map1 = map, frame_map2 = odom (의미 유지)
    pnh_.param<std::string>("frame_map1", frame_map1_, std::string("map"));
    pnh_.param<std::string>("frame_map2", frame_map2_, std::string("odom"));

    pnh_.param<double>("voxel_leaf", voxel_leaf_, 0.5);
    pnh_.param<double>("yaw_step_deg", yaw_step_deg_, 5.0);
    pnh_.param<double>("ndt_res", ndt_res_, 1.0);
    pnh_.param<int>("ndt_iter", ndt_iter_, 30);
    pnh_.param<double>("ndt_step_size", ndt_step_size_, 0.1);
    pnh_.param<double>("ndt_eps", ndt_eps_, 1e-3);

    // [ADD] publish aligned cloud topic (latched)
    pnh_.param<std::string>("aligned_topic", aligned_topic_, std::string("/map2_aligned"));

    if (map1_pcd_.empty())
      throw std::runtime_error("map1_pcd empty");

    map1_raw_.reset(new CloudT);
    map1_.reset(new CloudT);
    map2_raw_.reset(new CloudT);
    map2_.reset(new CloudT);

    loadAndPrepMap1();

    // [ADD] aligned cloud publisher (latched=true)
    map2_aligned_pub_ = nh_.advertise<sensor_msgs::PointCloud2>(aligned_topic_, 1, true);

    map2_sub_ = nh_.subscribe(map2_topic_, 1, &MapMapAligner::map2Cb, this);

    ROS_INFO("map_map_aligner started. map1_pcd=%s, map2_topic=%s, aligned_topic=%s",
             map1_pcd_.c_str(), map2_topic_.c_str(), aligned_topic_.c_str());
  }

private:
  void map2Cb(const sensor_msgs::PointCloud2ConstPtr& msg)
  {
    pcl::fromROSMsg(*msg, *map2_raw_);
    prepCloud(map2_raw_, map2_);
    computeAndPublishTF();
  }

  void loadAndPrepMap1()
  {
    if (pcl::io::loadPCDFile<PointT>(map1_pcd_, *map1_raw_) != 0)
      throw std::runtime_error("load map1 failed");
    prepCloud(map1_raw_, map1_);
    ROS_INFO("Loaded map1 PCD (%zu points after prep).", map1_->size());
  }

  void prepCloud(const CloudT::Ptr& in, CloudT::Ptr& out)
  {
    pcl::VoxelGrid<PointT> vg;
    vg.setLeafSize(static_cast<float>(voxel_leaf_),
                   static_cast<float>(voxel_leaf_),
                   static_cast<float>(voxel_leaf_));
    vg.setInputCloud(in);

    CloudT::Ptr tmp(new CloudT);
    vg.filter(*tmp);

    out.reset(new CloudT);
    out->reserve(tmp->size());
    for (const auto& p : tmp->points)
    {
      PointT q = p;
      q.z = 0.0f;  // 2D projection
      out->points.push_back(q);
    }
    out->width = out->points.size();
    out->height = 1;
    out->is_dense = true;
  }

  static Eigen::Matrix4f yawInit(float yaw)
  {
    Eigen::Matrix4f T = Eigen::Matrix4f::Identity();
    float c = std::cos(yaw), s = std::sin(yaw);
    T(0,0)=c; T(0,1)=-s;
    T(1,0)=s; T(1,1)=c;
    return T;
  }

  void computeAndPublishTF()
  {
    pcl::NormalDistributionsTransform<PointT, PointT> ndt;
    ndt.setTransformationEpsilon(ndt_eps_);
    ndt.setStepSize(ndt_step_size_);
    ndt.setResolution(ndt_res_);
    ndt.setMaximumIterations(ndt_iter_);

    // target = map1, source = map2
    ndt.setInputTarget(map1_);

    double best_score = std::numeric_limits<double>::max();
    Eigen::Matrix4f best_T = Eigen::Matrix4f::Identity();
    CloudT aligned;

    const double step = yaw_step_deg_ * M_PI / 180.0;
    for (double yaw=-M_PI; yaw<M_PI; yaw+=step)
    {
      ndt.setInputSource(map2_);
      ndt.align(aligned, yawInit(static_cast<float>(yaw)));
      if (ndt.hasConverged() && ndt.getFitnessScore() < best_score)
      {
        best_score = ndt.getFitnessScore();
        best_T = ndt.getFinalTransformation();
      }
    }

    ndt.setInputSource(map2_);
    ndt.align(aligned, best_T);
    if (!ndt.hasConverged())
    {
      ROS_WARN("NDT did not converge.");
      return;
    }

    Eigen::Matrix4f T = ndt.getFinalTransformation();

    // ============================
    // [ORIGINAL B MODE]
    // 같은 T를 odom -> map 으로 그대로 TF publish
    // (좌표 해석만 바뀌고, 점 자체는 안 바뀜)
    //
    // publishStaticTF(frame_map2_, frame_map1_, T);
    // ============================

    // =====================================================
    // [MODIFIED: POINT TRANSFORM + PUBLISH]
    // map2_cloud(odom frame) 점들을 T로 직접 변환해서 map frame cloud로 publish
    // =====================================================
    CloudT map2_aligned_cloud;
    pcl::transformPointCloud(*map2_raw_, map2_aligned_cloud, T);

    sensor_msgs::PointCloud2 msg_out;
    pcl::toROSMsg(map2_aligned_cloud, msg_out);
    msg_out.header.stamp = ros::Time::now();
    msg_out.header.frame_id = frame_map1_;  // "map" 기준으로 고정
    map2_aligned_pub_.publish(msg_out);

    double yaw = std::atan2(T(1,0), T(0,0));
    ROS_INFO("ALIGN T (odom->map, APPLY TO POINTS): x=%.3f y=%.3f yaw=%.2f deg | published: %s",
             T(0,3), T(1,3), yaw * 180.0 / M_PI, aligned_topic_.c_str());
  }

  void publishStaticTF(const std::string& parent,
                       const std::string& child,
                       const Eigen::Matrix4f& T)
  {
    geometry_msgs::TransformStamped tfm;
    tfm.header.stamp = ros::Time::now();
    tfm.header.frame_id = parent;
    tfm.child_frame_id  = child;

    tfm.transform.translation.x = T(0,3);
    tfm.transform.translation.y = T(1,3);
    tfm.transform.translation.z = 0.0;

    double yaw = std::atan2(T(1,0), T(0,0));
    tf2::Quaternion q;
    q.setRPY(0,0,yaw);
    tfm.transform.rotation.x = q.x();
    tfm.transform.rotation.y = q.y();
    tfm.transform.rotation.z = q.z();
    tfm.transform.rotation.w = q.w();

    static tf2_ros::StaticTransformBroadcaster br;
    br.sendTransform(tfm);
  }

private:
  ros::NodeHandle nh_, pnh_;
  ros::Subscriber map2_sub_;

  // [ADD] publisher
  ros::Publisher map2_aligned_pub_;
  std::string aligned_topic_;

  std::string map1_pcd_, map2_topic_, frame_map1_, frame_map2_;
  double voxel_leaf_, yaw_step_deg_, ndt_res_, ndt_step_size_, ndt_eps_;
  int ndt_iter_;

  CloudT::Ptr map1_raw_, map1_, map2_raw_, map2_;
};

int main(int argc, char** argv)
{
  ros::init(argc, argv, "map_map_aligner");
  ros::NodeHandle nh, pnh("~");
  try
  {
    MapMapAligner node(nh, pnh);
    ros::spin();
  }
  catch (const std::exception& e)
  {
    ROS_FATAL("Exception: %s", e.what());
    return 1;
  }
  return 0;
}
