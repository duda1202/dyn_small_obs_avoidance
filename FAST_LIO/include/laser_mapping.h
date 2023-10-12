/* 
© 2022 Robotics 88
Author: Erin Linebarger <erin@robotics88.com> 
*/

#ifndef LASER_MAPPING_H_
#define LASER_MAPPING_H_

#include <ros/ros.h>

#include <omp.h>
#include <mutex>
#include <math.h>
#include <fstream>
#include <csignal>
#include <unistd.h>
#include <Python.h>
#include <so3_math.h>
#include <Eigen/Core>
#include <opencv2/opencv.hpp>
#include <common_lib.h>
#include <kd_tree/kd_tree.h>
#include "IMU_Processing.hpp"
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <opencv2/core/eigen.hpp>
#include <visualization_msgs/Marker.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/io/pcd_io.h>
#include <sensor_msgs/PointCloud2.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>
#include <fast_lio/States.h>
#include <geometry_msgs/Vector3.h>

#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/time_synchronizer.h>

namespace laserMapping
{
/**
 * @class LaserMapping
 * @brief Takes a processed (leveled) pointcloud and registers it to ongoing map
 */
class LaserMapping
{
public:
  LaserMapping(ros::NodeHandle& node);
  ~LaserMapping(){};

private:
    ros::NodeHandle param_nh_;
    ros::NodeHandle nh_;

    int iterCount;
    int NUM_MAX_ITERATIONS;
    int INIT_TIME;
    double LASER_POINT_COV;
    int NUM_MATCH_POINTS;
    int FOV_RANGE;  // range of FOV = FOV_RANGE * cube_len
    int laserCloudCenWidth;
    int laserCloudCenHeight;
    int laserCloudCenDepth;
    double filter_size_map_min;
    bool dense_map_en;

    int laserCloudValidNum;
    int laserCloudSelNum;

    const int laserCloudWidth  = 48;
    const int laserCloudHeight = 48;
    const int laserCloudDepth  = 48;
    const int laserCloudNum = laserCloudWidth * laserCloudHeight * laserCloudDepth;

    std::string uav_frame_;
    std::string map_frame_;

    /// IMU relative variables
    std::mutex mtx_buffer; // TODO I dont think this is needed in this version
    // std::condition_variable sig_buffer;
    bool lidar_pushed = false;
    bool flg_exit = false;
    bool flg_reset = false;
    bool flg_map_inited = false;
    bool first_run_ = true;

    /// Buffers for measurements
    double cube_len = 20.0;
    double lidar_end_time = 0.0;
    double last_timestamp_lidar = -1;
    double last_timestamp_imu   = -1;
    double HALF_FOV_COS = 0.0;
    double res_mean_last = 0.05;
    double total_distance = 0.0;
    // auto   position_last  = Zero3d;
    double copy_time, readd_time;

    // Synced cb
    typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::PointCloud2, sensor_msgs::Imu> MySyncPolicy;
    typedef message_filters::Synchronizer<MySyncPolicy> syncPolicy;
    boost::shared_ptr<syncPolicy> sync_;
    message_filters::Subscriber<sensor_msgs::PointCloud2> lidar_subscriber_;
    message_filters::Subscriber<sensor_msgs::Imu> imu_subscriber_;

    // Publishers
    ros::Publisher pubLaserCloudFullRes;
    ros::Publisher pubLaserCloudEffect;
    ros::Publisher pubLaserCloudMap;
    ros::Publisher pubOdomAftMapped;
    ros::Publisher pubVisionPose;
    ros::Publisher pubPath;

    //surf feature in map
    PointCloudXYZI::Ptr featsFromMap; //(new PointCloudXYZI());
    PointCloudXYZI::Ptr cube_points_add;
    std::vector<BoxPointType> cub_needrm;
    std::vector<BoxPointType> cub_needad;

    //all points
    // std::vector<PointCloudXYZI::Ptr> laserCloudFullRes2;
    // std::vector<PointCloudXYZI::Ptr> featsArray;//[laserCloudNum];
    // std::vector<bool>                _last_inFOV;//[laserCloudNum];
    // std::vector<bool>                cube_updated;//[laserCloudNum];
    // std::vector<int> laserCloudValidInd;//[laserCloudNum];
    // pcl::PointCloud<pcl::PointXYZI>::Ptr laserCloudFullResColor;

    // KD tree
    KD_TREE ikdtree;

    Eigen::Vector3f XAxisPoint_body;
    Eigen::Vector3f XAxisPoint_world;

    //estimator inputs and output;
    MeasureGroup Measures;
    StatesGroup  state;

    void pointBodyToWorld(PointType const * const pi, PointType * const po);

    template<typename T>
    void pointBodyToWorld(const Eigen::Matrix<T, 3, 1> &pi, Eigen::Matrix<T, 3, 1> &po);

    void rgbPointBodyToWorld(PointType const * const pi, pcl::PointXYZI * const po);

    int cubeInd(const int &i, const int &j, const int &k);

    bool centerinFOV(Eigen::Vector3f cube_p);

    bool cornerinFOV(Eigen::Vector3f cube_p);

    void lasermapFovSegment(std::vector<PointCloudXYZI::Ptr> &featsArray);

    void dataProcessing();

    void lidarImuSyncedCallback(const sensor_msgs::PointCloud2::ConstPtr &lidar_msg, const sensor_msgs::Imu::ConstPtr &imu_msg);

};
}

#endif