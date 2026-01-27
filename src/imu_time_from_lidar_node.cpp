#include <ros/ros.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>

class ImuTimeFromLidar
{
public:
  ImuTimeFromLidar(ros::NodeHandle& nh)
  {
    nh.param<std::string>("imu_in", imu_in_, "/vectornav/IMU");
    nh.param<std::string>("lidar_in", lidar_in_, "/velodyne_points");
    nh.param<std::string>("imu_out", imu_out_, "/imu/data");
    nh.param<double>("imu_rate", imu_rate_, 40.0);

    dt_ = ros::Duration(1.0 / imu_rate_);

    imu_time_initialized_ = false;
    lidar_time_initialized_ = false;

    pub_imu_ = nh.advertise<sensor_msgs::Imu>(imu_out_, 500);
    sub_imu_ = nh.subscribe(imu_in_, 500, &ImuTimeFromLidar::imuCallback, this);
    sub_lidar_ = nh.subscribe(lidar_in_, 5, &ImuTimeFromLidar::lidarCallback, this);

    ROS_INFO("[imu_time_from_lidar] started (imu_rate = %.2f Hz)", imu_rate_);
  }

private:
  void lidarCallback(const sensor_msgs::PointCloud2ConstPtr& msg)
  {
    if (!lidar_time_initialized_)
    {
      lidar_t0_ = msg->header.stamp;
      lidar_time_initialized_ = true;
      ROS_INFO("[imu_time_from_lidar] LiDAR time reference set: %.6f",
               lidar_t0_.toSec());
    }
  }

  void imuCallback(const sensor_msgs::ImuConstPtr& msg)
  {
    if (!lidar_time_initialized_)
      return;  // LiDAR 기준 없으면 IMU 버림

    sensor_msgs::Imu out = *msg;

    if (!imu_time_initialized_)
    {
      imu_time_ = lidar_t0_;
      imu_time_initialized_ = true;
    }
    else
    {
      imu_time_ += dt_;
    }

    out.header.stamp = imu_time_;
    pub_imu_.publish(out);
  }

  // ROS
  ros::Subscriber sub_imu_;
  ros::Subscriber sub_lidar_;
  ros::Publisher  pub_imu_;

  // Params
  std::string imu_in_;
  std::string lidar_in_;
  std::string imu_out_;
  double imu_rate_;
  ros::Duration dt_;

  // Time state
  bool lidar_time_initialized_;
  bool imu_time_initialized_;
  ros::Time lidar_t0_;
  ros::Time imu_time_;
};

int main(int argc, char** argv)
{
  ros::init(argc, argv, "imu_time_from_lidar_node");
  ros::NodeHandle nh("~");

  ImuTimeFromLidar node(nh);
  ros::spin();

  return 0;
}
