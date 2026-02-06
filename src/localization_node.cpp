#include <ros/ros.h>

#include <sensor_msgs/PointCloud2.h>
#include <visualization_msgs/MarkerArray.h>

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/common/common.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/PointIndices.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
#include <tf2_eigen/tf2_eigen.h>
#include <pcl_ros/transforms.h>


#include <geometry_msgs/Point.h>

#include <Eigen/Dense>

#include <mutex>
#include <vector>
#include <string>

// ======================================================
//  Localization Node (Trajectory Anchor Builder)
// ======================================================

class LocalizationNode
{
public:
    using PointT = pcl::PointXYZI;

    LocalizationNode(ros::NodeHandle& nh, ros::NodeHandle& pnh)
        : nh_(nh), pnh_(pnh)
    {
        // ----------------------------
        // Parameters
        // ----------------------------
        pnh_.param<std::string>("global_topic", global_topic_, "/global_map_pcd");
        pnh_.param<std::string>("trajectory_topic", traj_topic_, "/trajectory_pcd");
        pnh_.param<std::string>("fixed_frame", fixed_frame_, "map");

        // anchor_min_dist > 0  : downsampling ON
        // anchor_min_dist <= 0 : downsampling OFF
        pnh_.param<double>("anchor_min_dist", anchor_min_dist_, 2.0);

        pnh_.param<bool>("publish_anchor_cloud", publish_anchor_cloud_, true);
        pnh_.param<bool>("publish_anchor_markers", publish_anchor_markers_, true);

        // submap extraction
        pnh_.param<double>("submap_radius", submap_radius_, 25.0);
        ROS_INFO("[LOC] submap_radius = %.2f m", submap_radius_);


        // ----------------------------
        // Publishers
        // ----------------------------
        pub_anchor_cloud_ =
            nh_.advertise<sensor_msgs::PointCloud2>("/anchor_pcd", 1, true);

        pub_anchor_markers_ =
            nh_.advertise<visualization_msgs::MarkerArray>("/anchor_markers", 1, true);

        pub_submap_cloud_ =
            nh_.advertise<sensor_msgs::PointCloud2>("/submap_pcd", 1, true);

        timer_ = nh_.createTimer(
            ros::Duration(0.1),   // 10 Hz
            &LocalizationNode::localizationLoop,
            this
        );

        // ----------------------------
        // Subscribers (one-shot)
        // ----------------------------
        sub_global_ =
            nh_.subscribe(global_topic_, 1, &LocalizationNode::globalCallback, this);

        sub_traj_ =
            nh_.subscribe(traj_topic_, 1, &LocalizationNode::trajectoryCallback, this);

        sub_local_ = nh_.subscribe(
            "/lio_sam/mapping/cloud_registered",
            1,
            &LocalizationNode::localScanCallback,
            this
        );

        ROS_INFO("[LOC] localization_node started");
        ROS_INFO("[LOC] global_topic    : %s", global_topic_.c_str());
        ROS_INFO("[LOC] trajectory_topic: %s", traj_topic_.c_str());
        ROS_INFO("[LOC] anchor_min_dist : %.3f", anchor_min_dist_);
    }

private:
    // ==================================================
    // Callbacks
    // ==================================================

    void globalCallback(const sensor_msgs::PointCloud2ConstPtr& msg)
    {
        std::lock_guard<std::mutex> lock(mtx_);
        if (global_received_) return;

        pcl::PointCloud<PointT> tmp;
        pcl::fromROSMsg(*msg, tmp);

        global_map_.reset(new pcl::PointCloud<PointT>(tmp));
        pcl::getMinMax3D(*global_map_, global_min_, global_max_);

        global_received_ = true;

        // KD-tree build once
        kdtree_global_.setInputCloud(global_map_);

        ROS_INFO("[LOC] Global map received: %lu points", global_map_->size());
        finalizeIfReady();
    }

    void trajectoryCallback(const sensor_msgs::PointCloud2ConstPtr& msg)
    {
        std::lock_guard<std::mutex> lock(mtx_);
        if (traj_received_) return;

        pcl::PointCloud<PointT> tmp;
        pcl::fromROSMsg(*msg, tmp);

        traj_cloud_.reset(new pcl::PointCloud<PointT>(tmp));
        pcl::getMinMax3D(*traj_cloud_, traj_min_, traj_max_);

        traj_received_ = true;

        ROS_INFO("[LOC] Trajectory received: %lu points", traj_cloud_->size());
        finalizeIfReady();
    }

    // ==================================================
    // Finalization
    // ==================================================

    void finalizeIfReady()
    {
        if (finalized_) return;
        if (!global_received_ || !traj_received_) return;

        buildAnchorsFromTrajectory();

        if (publish_anchor_cloud_)
            publishAnchorCloud();

        if (publish_anchor_markers_)
            publishAnchorMarkers();

        bboxSanityCheck();

        finalized_ = true;

        ROS_INFO("[LOC] Finalized. Anchors: %lu", anchors_.size());

        if (!anchors_.empty())
        {
            test_submap_ = extractSubmapAroundAnchor(anchors_[0]);
            ROS_INFO("[LOC] Test submap size (anchor 0): %lu points",
                    test_submap_->size());
            
            publishSubmapCloud(test_submap_);
        }
    }

    // ==================================================
    // Anchor generation
    // ==================================================

    void buildAnchorsFromTrajectory()
    {
        anchors_.clear();

        if (!traj_cloud_ || traj_cloud_->empty())
        {
            ROS_WARN("[LOC] Trajectory cloud empty");
            return;
        }

        // ------------------------------
        // OFF: use all trajectory points
        // ------------------------------
        if (anchor_min_dist_ <= 0.0)
        {
            for (const auto& pt : traj_cloud_->points)
            {
                anchors_.emplace_back(pt.x, pt.y, pt.z);
            }

            ROS_INFO("[LOC] Anchor downsampling OFF (%lu anchors)",
                     anchors_.size());
            return;
        }

        // ------------------------------
        // ON: distance-based sampling
        // ------------------------------
        const double minDist2 = anchor_min_dist_ * anchor_min_dist_;
        Eigen::Vector3f last(1e9f, 1e9f, 1e9f);

        for (const auto& pt : traj_cloud_->points)
        {
            Eigen::Vector3f cur(pt.x, pt.y, pt.z);
            if (anchors_.empty() || (cur - last).squaredNorm() >= minDist2)
            {
                anchors_.push_back(cur);
                last = cur;
            }
        }

        ROS_INFO("[LOC] Anchor downsampling ON (%lu anchors, min_dist=%.2f)",
                 anchors_.size(), anchor_min_dist_);
    }

    // ==================================================
    // Publish for RViz
    // ==================================================

    void publishAnchorCloud()
    {
        pcl::PointCloud<PointT> cloud;
        cloud.reserve(anchors_.size());

        for (size_t i = 0; i < anchors_.size(); ++i)
        {
            PointT p;
            p.x = anchors_[i].x();
            p.y = anchors_[i].y();
            p.z = anchors_[i].z();
            p.intensity = static_cast<float>(i);
            cloud.push_back(p);
        }

        sensor_msgs::PointCloud2 msg;
        pcl::toROSMsg(cloud, msg);
        msg.header.stamp = ros::Time::now();
        msg.header.frame_id = fixed_frame_;

        pub_anchor_cloud_.publish(msg);
    }

    void publishAnchorMarkers()
    {
        visualization_msgs::MarkerArray arr;
        visualization_msgs::Marker m;

        m.header.frame_id = fixed_frame_;
        m.header.stamp = ros::Time::now();
        m.ns = "anchors";
        m.id = 0;
        m.type = visualization_msgs::Marker::SPHERE_LIST;
        m.action = visualization_msgs::Marker::ADD;

        m.scale.x = 0.3;
        m.scale.y = 0.3;
        m.scale.z = 0.3;

        m.color.a = 1.0;
        m.color.r = 1.0;
        m.color.g = 0.3;
        m.color.b = 0.0;

        for (const auto& a : anchors_)
        {
            geometry_msgs::Point p;
            p.x = a.x();
            p.y = a.y();
            p.z = a.z();
            m.points.push_back(p);
        }

        arr.markers.push_back(m);
        pub_anchor_markers_.publish(arr);
    }

    // ==================================================
    // Sanity check
    // ==================================================

    void bboxSanityCheck()
    {
        bool overlap_x = !(traj_max_.x() < global_min_.x() ||
                           traj_min_.x() > global_max_.x());
        bool overlap_y = !(traj_max_.y() < global_min_.y() ||
                           traj_min_.y() > global_max_.y());
        bool overlap_z = !(traj_max_.z() < global_min_.z() ||
                           traj_min_.z() > global_max_.z());

        if (overlap_x && overlap_y && overlap_z)
            ROS_INFO("[LOC] BBox overlap OK (trajectory vs global)");
        else
            ROS_WARN("[LOC] BBox NOT overlapping (check frame / scale)");
    }

    pcl::PointCloud<PointT>::Ptr
    extractSubmapAroundAnchor(const Eigen::Vector3f& anchor)
    {
        pcl::PointCloud<PointT>::Ptr submap(
            new pcl::PointCloud<PointT>());

        if (!global_map_ || global_map_->empty())
            return submap;

        PointT searchPoint;
        searchPoint.x = anchor.x();
        searchPoint.y = anchor.y();
        searchPoint.z = anchor.z();

        std::vector<int> indices;
        std::vector<float> sqr_dists;

        kdtree_global_.radiusSearch(
            searchPoint,
            submap_radius_,
            indices,
            sqr_dists
        );

        submap->reserve(indices.size());
        for (int idx : indices)
            submap->push_back(global_map_->points[idx]);

        return submap;
    }

    void publishSubmapCloud(const pcl::PointCloud<PointT>::Ptr& submap)
    {
        if (!submap || submap->empty())
            return;

        sensor_msgs::PointCloud2 msg;
        pcl::toROSMsg(*submap, msg);
        msg.header.stamp = ros::Time::now();
        msg.header.frame_id = fixed_frame_;

        pub_submap_cloud_.publish(msg);
    }

    pcl::PointCloud<PointT>::Ptr
    extractFloorCandidates(const pcl::PointCloud<PointT>::Ptr& submap)
    {
        pcl::PointCloud<PointT>::Ptr floor(new pcl::PointCloud<PointT>());
        if (!submap || submap->empty()) return floor;

        std::vector<float> zs;
        zs.reserve(submap->size());
        for (const auto& p : submap->points)
            zs.push_back(p.z);

        std::nth_element(
            zs.begin(),
            zs.begin() + zs.size() / 10,
            zs.end());

        float z_thresh = zs[zs.size() / 10] + 0.25f;

        for (const auto& p : submap->points)
            if (p.z < z_thresh)
                floor->push_back(p);

        return floor;
    }

    bool estimateFloorPlane(
        const pcl::PointCloud<PointT>::Ptr& floor_cloud,
        pcl::ModelCoefficients::Ptr& coeffs,
        pcl::PointIndices::Ptr& inliers)
    {
        if (!floor_cloud || floor_cloud->size() < 50)
            return false;

        pcl::SACSegmentation<PointT> seg;
        seg.setOptimizeCoefficients(true);
        seg.setModelType(pcl::SACMODEL_PLANE);
        seg.setMethodType(pcl::SAC_RANSAC);
        seg.setDistanceThreshold(0.08);
        seg.setInputCloud(floor_cloud);

        seg.segment(*inliers, *coeffs);
        return !inliers->indices.empty();
    }

    void localScanCallback(const sensor_msgs::PointCloud2ConstPtr& msg)
    {
        std::lock_guard<std::mutex> lock(mtx_);

        sensor_msgs::PointCloud2 msg_tf;
        try {
            pcl_ros::transformPointCloud(
                fixed_frame_,   // target frame
                *msg,           // input cloud
                msg_tf,         // output cloud
                tf_buffer_      // tf2 buffer
            );
        } catch (const tf2::TransformException& ex) {
            ROS_WARN_THROTTLE(1.0, "[LOC] TF transform failed: %s", ex.what());
            return;
        }

        pcl::PointCloud<PointT> tmp;
        pcl::fromROSMsg(msg_tf, tmp);

        local_cloud_.reset(new pcl::PointCloud<PointT>(tmp));
        local_received_ = true;
    }


    void localizationLoop(const ros::TimerEvent&)
    {
        std::lock_guard<std::mutex> lock(mtx_);

        // 1. 준비 안 됐으면 탈출
        if (!finalized_) return;
        if (!local_received_) return;
        if (!test_submap_ || test_submap_->empty()) return;

        // 2. local scan 바닥 후보
        auto floor_local = extractFloorCandidates(local_cloud_);

        pcl::ModelCoefficients::Ptr coeffs_l(new pcl::ModelCoefficients);
        pcl::PointIndices::Ptr inliers_l(new pcl::PointIndices);

        if (!estimateFloorPlane(floor_local, coeffs_l, inliers_l))
            return;

        // 3. submap 바닥 후보
        auto floor_submap = extractFloorCandidates(test_submap_);

        pcl::ModelCoefficients::Ptr coeffs_s(new pcl::ModelCoefficients);
        pcl::PointIndices::Ptr inliers_s(new pcl::PointIndices);

        if (!estimateFloorPlane(floor_submap, coeffs_s, inliers_s))
            return;

        // 4. 여기서부터 plane-to-plane 정합
        Eigen::Vector3f n_local(
            coeffs_l->values[0],
            coeffs_l->values[1],
            coeffs_l->values[2]);

        Eigen::Vector3f n_submap(
            coeffs_s->values[0],
            coeffs_s->values[1],
            coeffs_s->values[2]);

        n_local.normalize();
        n_submap.normalize();
        float cos_angle = n_local.dot(n_submap);
        cos_angle = std::max(-1.0f, std::min(1.0f, cos_angle));

        float angle_deg = std::acos(cos_angle) * 180.0f / M_PI;

        ROS_INFO_THROTTLE(
            1.0,
            "[FLOOR] normal angle diff = %.3f deg",
            angle_deg
        );

    }


private:
    // ROS
    ros::NodeHandle nh_;
    ros::NodeHandle pnh_;

    ros::Subscriber sub_global_;
    ros::Subscriber sub_traj_;
    ros::Subscriber sub_local_;

    ros::Publisher pub_anchor_cloud_;
    ros::Publisher pub_anchor_markers_;

    ros::Publisher pub_submap_cloud_;

    ros::Timer timer_;

    tf2_ros::Buffer tf_buffer_;
    tf2_ros::TransformListener tf_listener_{tf_buffer_};

    // Data
    pcl::PointCloud<PointT>::Ptr global_map_;
    pcl::PointCloud<PointT>::Ptr traj_cloud_;
    pcl::KdTreeFLANN<PointT> kdtree_global_;
    pcl::PointCloud<PointT>::Ptr test_submap_;
    pcl::PointCloud<PointT>::Ptr local_cloud_;
    
    Eigen::Vector4f global_min_, global_max_;
    Eigen::Vector4f traj_min_, traj_max_;

    // std::vector<Eigen::Vector3f> anchors_;
    std::vector<Eigen::Vector3f,
        Eigen::aligned_allocator<Eigen::Vector3f>> anchors_;

    // State
    bool global_received_ = false;
    bool traj_received_ = false;
    bool finalized_ = false;
    bool local_received_ = false;

    // Params
    std::string global_topic_;
    std::string traj_topic_;
    std::string fixed_frame_;

    double anchor_min_dist_ = 2.0;
    double submap_radius_ = 25.0;
    bool publish_anchor_cloud_ = true;
    bool publish_anchor_markers_ = true;

    std::mutex mtx_;
};

// ======================================================
// Main
// ======================================================

int main(int argc, char** argv)
{
    ros::init(argc, argv, "lio_sam_localization");
    ros::NodeHandle nh;
    ros::NodeHandle pnh("~");

    LocalizationNode node(nh, pnh);
    ros::spin();

    return 0;
}
