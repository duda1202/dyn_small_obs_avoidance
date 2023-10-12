/* 
© 2022 Robotics 88
Author: Erin Linebarger <erin@robotics88.com> 
*/

#include <laser_mapping.h>

#include <thread>

namespace laserMapping
{
LaserMapping::LaserMapping(ros::NodeHandle& node)
  : param_nh_("~")
  , nh_(node)
  , iterCount(0)
  , NUM_MAX_ITERATIONS(0)
  , INIT_TIME(0)
  , LASER_POINT_COV(.0015)
  , NUM_MATCH_POINTS(5)
  , FOV_RANGE(3)
  , laserCloudCenWidth(24)
  , laserCloudCenHeight(24)
  , laserCloudCenDepth(24)
  , filter_size_map_min(0.2)
  , dense_map_en(true)
  , laserCloudValidNum(0)
  , laserCloudSelNum(0)
  , uav_frame_("base_link")
  , map_frame_("map")
  , XAxisPoint_body(LIDAR_SP_LEN, 0.0, 0.0)
  , XAxisPoint_world(LIDAR_SP_LEN, 0.0, 0.0)
{
    ROS_INFO("LaserMapping:: entered constrx");
    param_nh_.param<std::string>("uav_frame", uav_frame_, uav_frame_);
	param_nh_.param<std::string>("map_frame", map_frame_, map_frame_);
	std::string proc_cloud_topic = "/laser_cloud_flat";
	param_nh_.param<std::string>("proc_cloud_topic", proc_cloud_topic, proc_cloud_topic);
	std::string reg_cloud_topic = "/cloud_registered";
	param_nh_.param<std::string>("reg_cloud_topic", reg_cloud_topic, reg_cloud_topic);
	std::string imu_topic = "/livox/imu";
	param_nh_.param<std::string>("imu_topic", imu_topic, imu_topic);
	std::string odom_topic = "/mavros/odometry/out";
	param_nh_.param<std::string>("odom_topic", odom_topic, odom_topic);
    std::string vision_topic = "/mavros/vision_pose/pose";
	param_nh_.param<std::string>("vision_topic", vision_topic, vision_topic);

    lidar_subscriber_.subscribe(param_nh_, proc_cloud_topic, 10);
    imu_subscriber_.subscribe(param_nh_, imu_topic, 10);
    sync_.reset(new syncPolicy(MySyncPolicy(10), lidar_subscriber_, imu_subscriber_));
    sync_->registerCallback(boost::bind(&LaserMapping::lidarImuSyncedCallback, this, _1, _2));

    pubLaserCloudFullRes = nh_.advertise<sensor_msgs::PointCloud2>
            (reg_cloud_topic, 100);
    pubLaserCloudEffect  = nh_.advertise<sensor_msgs::PointCloud2>
            ("/cloud_effected", 100);
    pubLaserCloudMap = nh_.advertise<sensor_msgs::PointCloud2>
            ("/Laser_map", 100);
    pubOdomAftMapped = nh_.advertise<nav_msgs::Odometry> 
            (odom_topic, 10);
    pubVisionPose = nh_.advertise<geometry_msgs::PoseStamped> 
            (vision_topic, 10);
    pubPath          = nh_.advertise<nav_msgs::Path> 
            ("/path", 10);

    ROS_INFO("LaserMapping:: got params and set up subs, pubs");

    cv::Mat matA1(3, 3, CV_32F, cv::Scalar::all(0));
    cv::Mat matD1(1, 3, CV_32F, cv::Scalar::all(0));
    cv::Mat matV1(3, 3, CV_32F, cv::Scalar::all(0));
    cv::Mat matP(6, 6, CV_32F, cv::Scalar::all(0));

    PointType pointOri, pointSel, coeff;
    PointCloudXYZI::Ptr feats_undistort(new PointCloudXYZI());
    PointCloudXYZI::Ptr feats_down(new PointCloudXYZI());
    PointCloudXYZI::Ptr laserCloudOri(new PointCloudXYZI());
    PointCloudXYZI::Ptr coeffSel(new PointCloudXYZI());
    pcl::VoxelGrid<PointType> downSizeFilterSurf;
    pcl::VoxelGrid<PointType> downSizeFilterMap;

    /*** variables initialize ***/
    param_nh_.param("dense_map_enable",dense_map_en);
    param_nh_.param("max_iteration",NUM_MAX_ITERATIONS);
    // param_nh_.param("~fov_degree",fov_deg);
    // param_nh_.param("~filter_size_corner",filter_size_corner_min);
    // param_nh_.param("~filter_size_surf",filter_size_surf_min);
    param_nh_.param("filter_size_map",filter_size_map_min);
    param_nh_.param("cube_side_length",cube_len);
}

void LaserMapping::lidarImuSyncedCallback(const sensor_msgs::PointCloud2::ConstPtr &lidar_msg, const sensor_msgs::Imu::ConstPtr &imu_msg) {
    ROS_INFO("entered lidar IMU synced cb");
    mtx_buffer.lock();
    Measures.imu.clear();
    sensor_msgs::Imu::Ptr imu_ptr(new sensor_msgs::Imu(*imu_msg));
    Measures.imu.push_back(imu_ptr);

    Measures.lidar.reset(new PointCloudXYZI());
    Measures.lidar_beg_time = lidar_msg->header.stamp.toSec();
    pcl::fromROSMsg(*lidar_msg, *(Measures.lidar));

    dataProcessing();
    mtx_buffer.unlock();
}

void LaserMapping::dataProcessing() {
    // if (first_run_) {
    //     ikdtree.init();
    //     first_run_ = false;
    // }
    ROS_INFO("LaserMapping::dataProcessing entered ");
    std::shared_ptr<ImuProcess> p_imu(new ImuProcess());
    PointType pointOri, pointSel, coeff;
    PointCloudXYZI::Ptr feats_undistort(new PointCloudXYZI());
    PointCloudXYZI::Ptr feats_down(new PointCloudXYZI());
    PointCloudXYZI::Ptr laserCloudOri(new PointCloudXYZI());
    PointCloudXYZI::Ptr coeffSel(new PointCloudXYZI());
    pcl::VoxelGrid<PointType> downSizeFilterSurf;
    pcl::VoxelGrid<PointType> downSizeFilterMap;
    double first_lidar_time, deltaT, deltaR;
    bool flg_EKF_inited = 0, flg_EKF_converged = 0;

    Eigen::Matrix<double,DIM_OF_STATES,DIM_OF_STATES> G, H_T_H, I_STATE;
    G.setZero();
    H_T_H.setZero();
    I_STATE.setIdentity();

    double t0,t1,t2,t3,t4,t5,match_start, match_time, solve_start, solve_time, pca_time, svd_time;
    match_time = 0;
    solve_time = 0;
    pca_time   = 0;
    svd_time   = 0;
    t0 = omp_get_wtime();

    p_imu->Process(Measures, state, feats_undistort);
    StatesGroup state_propagat(state);

    if (feats_undistort->empty() || (feats_undistort == NULL))
    {
        first_lidar_time = Measures.lidar_beg_time;
        std::cout<<"not ready for odometry"<<std::endl;
        return;
    }

    if ((Measures.lidar_beg_time - first_lidar_time) < INIT_TIME)
    {
        flg_EKF_inited = false;
        std::cout<<"||||||||||Initiallizing LiDar||||||||||"<<std::endl;
    }
    else
    {
        flg_EKF_inited = true;
    }
    
    /*** Compute the euler angle ***/
    Eigen::Vector3d euler_cur = RotMtoEuler(state.rot_end);

    //all points
    PointCloudXYZI::Ptr laserCloudFullRes2(new PointCloudXYZI());
    std::vector<PointCloudXYZI::Ptr> featsArray(laserCloudNum,0);
    bool cube_updated[laserCloudNum];
    pcl::PointCloud<pcl::PointXYZI>::Ptr laserCloudFullResColor(new pcl::PointCloud<pcl::PointXYZI>());
    
    /*** Segment the map in lidar FOV ***/
    lasermapFovSegment(featsArray);
    
    /*** downsample the features of new frame ***/
    downSizeFilterSurf.setInputCloud(feats_undistort);
    downSizeFilterSurf.filter(*feats_down);

    /*** initialize the map kdtree ***/
    if((feats_down->points.size() > 1) && (ikdtree.Root_Node == nullptr))
    {
        // std::vector<PointType> points_init = feats_down->points;
        ikdtree.set_downsample_param(filter_size_map_min);
        ikdtree.Build(feats_down->points);
        flg_map_inited = true;
        return;
    }

    if(ikdtree.Root_Node == nullptr)
    {
        flg_map_inited = false;
        std::cout<<"~~~~~~~Initiallize Map iKD-Tree Failed!"<<std::endl;
        return;
    }
    int featsFromMapNum = ikdtree.size();

    int feats_down_size = feats_down->points.size();
    std::cout<<"[ mapping ]: Raw feature num: "<<feats_undistort->points.size()<<" downsamp num "<<feats_down_size<<" Map num: "<<featsFromMapNum<<" laserCloudValidNum "<<laserCloudValidNum<<std::endl;

    /*** ICP and iterated Kalman filter update ***/
    PointCloudXYZI::Ptr coeffSel_tmpt (new PointCloudXYZI(*feats_down));
    PointCloudXYZI::Ptr feats_down_updated (new PointCloudXYZI(*feats_down));
    std::vector<double> res_last(feats_down_size, 1000.0); // initial

    if (featsFromMapNum >= 5)
    {
        t1 = omp_get_wtime();

        std::vector<bool> point_selected_surf(feats_down_size, true);
        std::vector<std::vector<int>> pointSearchInd_surf(feats_down_size);
        std::vector<PointVector> Nearest_Points(feats_down_size);
        
        int  rematch_num = 0;
        bool rematch_en = 0;
        flg_EKF_converged = 0;
        deltaR = 0.0;
        deltaT = 0.0;
        t2 = omp_get_wtime();
        
        for (iterCount = 0; iterCount < NUM_MAX_ITERATIONS; iterCount++) 
        {
            match_start = omp_get_wtime();
            laserCloudOri->clear();
            coeffSel->clear();

            /** closest surface search and residual computation **/
            omp_set_num_threads(4);
            #pragma omp parallel for
            for (int i = 0; i < feats_down_size; i++)
            {
                PointType &pointOri_tmpt = feats_down->points[i];
                PointType &pointSel_tmpt = feats_down_updated->points[i];

                /* transform to world frame */
                pointBodyToWorld(&pointOri_tmpt, &pointSel_tmpt);
                std::vector<float> pointSearchSqDis_surf;
            #ifdef USE_ikdtree
                auto &points_near = Nearest_Points[i];
            #else
                auto &points_near = pointSearchInd_surf[i];
            #endif
                
                if (iterCount == 0 || rematch_en)
                {
                    point_selected_surf[i] = true;
                    /** Find the closest surfaces in the map **/
                #ifdef USE_ikdtree
                    ikdtree.Nearest_Search(pointSel_tmpt, NUM_MATCH_POINTS, points_near, pointSearchSqDis_surf);
                #else
                    kdtreeSurfFromMap->nearestKSearch(pointSel_tmpt, NUM_MATCH_POINTS, points_near, pointSearchSqDis_surf);
                #endif
                    float max_distance = pointSearchSqDis_surf[NUM_MATCH_POINTS - 1];
                
                    if (max_distance > 3)
                    {
                        point_selected_surf[i] = false;
                    }
                }

                if (point_selected_surf[i] == false) continue;

                // match_time += omp_get_wtime() - match_start;

                double pca_start = omp_get_wtime();

                /// PCA (using minimum square method)
                cv::Mat matA0(NUM_MATCH_POINTS, 3, CV_32F, cv::Scalar::all(0));
                cv::Mat matB0(NUM_MATCH_POINTS, 1, CV_32F, cv::Scalar::all(-1));
                cv::Mat matX0(NUM_MATCH_POINTS, 1, CV_32F, cv::Scalar::all(0));
                
                for (int j = 0; j < NUM_MATCH_POINTS; j++)
                {
                    matA0.at<float>(j, 0) = points_near[j].x;
                    matA0.at<float>(j, 1) = points_near[j].y;
                    matA0.at<float>(j, 2) = points_near[j].z;
                }
    
                cv::solve(matA0, matB0, matX0, cv::DECOMP_QR);  //TODO

                float pa = matX0.at<float>(0, 0);
                float pb = matX0.at<float>(1, 0);
                float pc = matX0.at<float>(2, 0);
                float pd = 1;

                //ps is the norm of the plane norm_vec vector
                //pd is the distance from point to plane
                float ps = sqrt(pa * pa + pb * pb + pc * pc);
                pa /= ps;
                pb /= ps;
                pc /= ps;
                pd /= ps;

                bool planeValid = true;
                for (int j = 0; j < NUM_MATCH_POINTS; j++)
                {
                    if (fabs(pa * points_near[j].x +
                                pb * points_near[j].y +
                                pc * points_near[j].z + pd) > 0.1)
                    {
                        planeValid = false;
                        point_selected_surf[i] = false;
                        break;
                    }
                }

                if (planeValid) 
                {
                    //loss fuction
                    float pd2 = pa * pointSel_tmpt.x + pb * pointSel_tmpt.y + pc * pointSel_tmpt.z + pd;
                    //if(fabs(pd2) > 0.1) continue;
                    float s = 1 - 0.9 * fabs(pd2) / sqrt(sqrt(pointSel_tmpt.x * pointSel_tmpt.x + pointSel_tmpt.y * pointSel_tmpt.y + pointSel_tmpt.z * pointSel_tmpt.z));

                    if ((s > 0.85))// && ((std::abs(pd2) - res_last[i]) < 3 * res_mean_last))
                    {
                        // if(std::abs(pd2) > 5 * res_mean_last)
                        // {
                        //     point_selected_surf[i] = false;
                        //     res_last[i] = 0.0;
                        //     continue;
                        // }
                        point_selected_surf[i] = true;
                        coeffSel_tmpt->points[i].x = pa;
                        coeffSel_tmpt->points[i].y = pb;
                        coeffSel_tmpt->points[i].z = pc;
                        coeffSel_tmpt->points[i].intensity = pd2;
                        
                        // if(i%50==0) std::cout<<"s: "<<s<<"last res: "<<res_last[i]<<" current res: "<<std::abs(pd2)<<std::endl;
                        res_last[i] = std::abs(pd2);
                    }
                    else
                    {
                        point_selected_surf[i] = false;
                    }
                }

                pca_time += omp_get_wtime() - pca_start;
            }

            double total_residual = 0.0;
            laserCloudSelNum = 0;

            for (int i = 0; i < coeffSel_tmpt->points.size(); i++)
            {
                if (point_selected_surf[i] && (res_last[i] <= 2.0))
                {
                    laserCloudOri->push_back(feats_down->points[i]);
                    coeffSel->push_back(coeffSel_tmpt->points[i]);
                    total_residual += res_last[i];
                    laserCloudSelNum ++;
                }
            }

            res_mean_last = total_residual / laserCloudSelNum;
            std::cout << "[ mapping ]: Effective feature num: "<<laserCloudSelNum<<" res_mean_last "<<res_mean_last<<std::endl;

            match_time += omp_get_wtime() - match_start;
            solve_start = omp_get_wtime();
            
            /*** Computation of Measuremnt Jacobian matrix H and measurents vector ***/
            Eigen::MatrixXd Hsub(laserCloudSelNum, 6);
            Eigen::VectorXd meas_vec(laserCloudSelNum);
            Hsub.setZero();

            // omp_set_num_threads(4);
            // #pragma omp parallel for
            for (int i = 0; i < laserCloudSelNum; i++)
            {
                const PointType &laser_p  = laserCloudOri->points[i];
                Eigen::Vector3d point_this(laser_p.x, laser_p.y, laser_p.z);
                point_this += Lidar_offset_to_IMU;
                Eigen::Matrix3d point_crossmat;
                point_crossmat<<SKEW_SYM_MATRX(point_this);

                /*** get the normal vector of closest surface/corner ***/
                const PointType &norm_p = coeffSel->points[i];
                Eigen::Vector3d norm_vec(norm_p.x, norm_p.y, norm_p.z);

                /*** calculate the Measuremnt Jacobian matrix H ***/
                Eigen::Vector3d A(point_crossmat * state.rot_end.transpose() * norm_vec);
                Hsub.row(i) << VEC_FROM_ARRAY(A), norm_p.x, norm_p.y, norm_p.z;

                /*** Measuremnt: distance to the closest surface/corner ***/
                meas_vec(i) = - norm_p.intensity;
            }

            Eigen::Vector3d rot_add, t_add, v_add, bg_add, ba_add, g_add;
            Eigen::Matrix<double, DIM_OF_STATES, 1> solution;
            Eigen::MatrixXd K(DIM_OF_STATES, laserCloudSelNum);
            
            /*** Iterative Kalman Filter Update ***/
            if (!flg_EKF_inited)
            {
                /*** only run in initialization period ***/
                Eigen::MatrixXd H_init(Eigen::Matrix<double, 9, DIM_OF_STATES>::Zero());
                Eigen::MatrixXd z_init(Eigen::Matrix<double, 9, 1>::Zero());
                H_init.block<3,3>(0,0)  = Eigen::Matrix3d::Identity();
                H_init.block<3,3>(3,3)  = Eigen::Matrix3d::Identity();
                H_init.block<3,3>(6,15) = Eigen::Matrix3d::Identity();
                z_init.block<3,1>(0,0)  = - Log(state.rot_end);
                z_init.block<3,1>(0,0)  = - state.pos_end;

                auto H_init_T = H_init.transpose();
                auto &&K_init = state.cov * H_init_T * (H_init * state.cov * H_init_T + 0.0001 * Eigen::Matrix<double,9,9>::Identity()).inverse();
                solution      = K_init * z_init;

                solution.block<9,1>(0,0).setZero();
                state += solution;
                state.cov = (Eigen::MatrixXd::Identity(DIM_OF_STATES, DIM_OF_STATES) - K_init * H_init) * state.cov;
            }
            else
            {
                auto &&Hsub_T = Hsub.transpose();
                H_T_H.block<6,6>(0,0) = Hsub_T * Hsub;
                Eigen::Matrix<double, DIM_OF_STATES, DIM_OF_STATES> &&K_1 = \
                            (H_T_H + (state.cov / LASER_POINT_COV).inverse()).inverse();
                K = K_1.block<DIM_OF_STATES,6>(0,0) * Hsub_T;

                // solution = K * meas_vec;
                // state += solution;

                auto vec = state_propagat - state;
                solution = K * (meas_vec - Hsub * vec.block<6,1>(0,0));
                state = state_propagat + solution;

                rot_add = solution.block<3,1>(0,0);
                t_add   = solution.block<3,1>(3,0);

                flg_EKF_converged = false;

                if (((rot_add.norm() * 57.3 - deltaR) < 0.01) \
                    && ((t_add.norm() * 100 - deltaT) < 0.015))
                {
                    flg_EKF_converged = true;
                }

                deltaR = rot_add.norm() * 57.3;
                deltaT = t_add.norm() * 100;
            }

            euler_cur = RotMtoEuler(state.rot_end);

            #ifdef DEBUG_PRINT
            std::cout<<"update: R"<<euler_cur.transpose()*57.3<<" p "<<state.pos_end.transpose()<<" v "<<state.vel_end.transpose()<<" bg"<<state.bias_g.transpose()<<" ba"<<state.bias_a.transpose()<<std::endl;
            std::cout<<"dR & dT: "<<deltaR<<" "<<deltaT<<" res norm:"<<res_mean_last<<std::endl;
            #endif

            /*** Rematch Judgement ***/
            rematch_en = false;
            if (flg_EKF_converged || ((rematch_num == 0) && (iterCount == (NUM_MAX_ITERATIONS - 2))))
            {
                rematch_en = true;
                rematch_num ++;
                std::cout<<"rematch_num: "<<rematch_num<<std::endl;
            }

            /*** Convergence Judgements and Covariance Update ***/
            if (rematch_num >= 2 || (iterCount == NUM_MAX_ITERATIONS - 1))
            {
                if (flg_EKF_inited)
                {
                    /*** Covariance Update ***/
                    G.block<DIM_OF_STATES,6>(0,0) = K * Hsub;
                    state.cov = (I_STATE - G) * state.cov;
                    // total_distance += (state.pos_end - position_last).norm();
                    // position_last = state.pos_end;
                    
                    // std::cout<<"position: "<<state.pos_end.transpose()<<" total distance: "<<total_distance<<std::endl;
                }
                solve_time += omp_get_wtime() - solve_start;
                break;
            }
            solve_time += omp_get_wtime() - solve_start;
        }
        
        std::cout<<"[ mapping ]: iteration count: "<<iterCount+1<<std::endl;

        t3 = omp_get_wtime();

        /*** add new frame points to map ikdtree ***/
        PointVector points_history;
        ikdtree.acquire_removed_points(points_history);
        
        memset(cube_updated, 0, sizeof(cube_updated));
        
        for (int i = 0; i < points_history.size(); i++)
        {
            PointType &pointSel = points_history[i];

            int cubeI = int((pointSel.x + 0.5 * cube_len) / cube_len) + laserCloudCenWidth;
            int cubeJ = int((pointSel.y + 0.5 * cube_len) / cube_len) + laserCloudCenHeight;
            int cubeK = int((pointSel.z + 0.5 * cube_len) / cube_len) + laserCloudCenDepth;

            if (pointSel.x + 0.5 * cube_len < 0) cubeI--;
            if (pointSel.y + 0.5 * cube_len < 0) cubeJ--;
            if (pointSel.z + 0.5 * cube_len < 0) cubeK--;

            if (cubeI >= 0 && cubeI < laserCloudWidth &&
                    cubeJ >= 0 && cubeJ < laserCloudHeight &&
                    cubeK >= 0 && cubeK < laserCloudDepth) 
            {
                int cubeInd = cubeI + laserCloudWidth * cubeJ + laserCloudWidth * laserCloudHeight * cubeK;
                featsArray.at(cubeInd)->push_back(pointSel);
                
            }
        }

        // omp_set_num_threads(4);
        // #pragma omp parallel for
        for (int i = 0; i < feats_down_size; i++)
        {
            /* transform to world frame */
            pointBodyToWorld(&(feats_down->points[i]), &(feats_down_updated->points[i]));
        }
        t4 = omp_get_wtime();
        ikdtree.Add_Points(feats_down_updated->points, true);

        t5 = omp_get_wtime();
    }
        
    /******* Publish current frame points in world coordinates:  *******/
    laserCloudFullRes2->clear();
    *laserCloudFullRes2 = dense_map_en ? (*feats_undistort) : (* feats_down);

    int laserCloudFullResNum = laserCloudFullRes2->points.size();

    pcl::PointXYZI temp_point;
    laserCloudFullResColor->clear();
    {
    for (int i = 0; i < laserCloudFullResNum; i++)
    {
        rgbPointBodyToWorld(&laserCloudFullRes2->points[i], &temp_point);
        laserCloudFullResColor->push_back(temp_point);
    }

    sensor_msgs::PointCloud2 laserCloudFullRes3;
    pcl::toROSMsg(*laserCloudFullResColor, laserCloudFullRes3);
    laserCloudFullRes3.header.stamp = ros::Time::now();//.fromSec(last_timestamp_lidar);
    laserCloudFullRes3.header.frame_id = map_frame_;
    pubLaserCloudFullRes.publish(laserCloudFullRes3);
    }

    /******* Publish Effective points *******/
    {
    laserCloudFullResColor->clear();
    pcl::PointXYZI temp_point;
    for (int i = 0; i < laserCloudSelNum; i++)
    {
        rgbPointBodyToWorld(&laserCloudOri->points[i], &temp_point);
        laserCloudFullResColor->push_back(temp_point);
    }
    sensor_msgs::PointCloud2 laserCloudFullRes3;
    pcl::toROSMsg(*laserCloudFullResColor, laserCloudFullRes3);
    laserCloudFullRes3.header.stamp = ros::Time::now();//.fromSec(last_timestamp_lidar);
    laserCloudFullRes3.header.frame_id = map_frame_;
    pubLaserCloudEffect.publish(laserCloudFullRes3);
    }

    /******* Publish Maps:  *******/
    // sensor_msgs::PointCloud2 laserCloudMap;
    // pcl::toROSMsg(*featsFromMap, laserCloudMap);
    // laserCloudMap.header.stamp = ros::Time::now();//ros::Time().fromSec(last_timestamp_lidar);
    // laserCloudMap.header.frame_id = map_frame_;
    // pubLaserCloudMap.publish(laserCloudMap);

    /******* Publish Odometry ******/
    // // Transfer to Inertial frame
    // Eigen::Matrix3d R_to_inertial;
    // double G_norm = state.gravity.norm();
    // Eigen::Vector3d G_standard(0,0,-G_norm);
    // Eigen::Vector3d rot_axis = G_standard.cross(state.gravity) / G_norm / G_norm;
    // if(rot_axis.norm()<0.001)
    // {
    //     R_to_inertial = Exp(rot_axis);
    // }
    // else
    // {
    //     Eigen::Vector3d ang_axis = rot_axis / rot_axis.norm();
    //     double ang      = std::asin(rot_axis.norm());
    //     R_to_inertial = Exp(ang_axis, ang);
    // }

    // // Eigen::Vector3d UAV_offset_to_LiD(0.02, 0.02, 0.02);
    // Eigen::Vector3d UAV_offset_to_LiD(-0.05, -0.01, 0.00);
    // // Eigen::Vector3d UAV_offset_to_LiD(-0.00, -0.00, 0.00);
    // Eigen::Vector3d pos_uav = R_to_inertial*(state.pos_end + state.rot_end * UAV_offset_to_LiD);


    // //add transform from camera_init to UAV coordinate 
    // //初始化欧拉角(Z-Y-X，即RPY, 先绕x轴roll,再绕y轴pitch,最后绕z轴yaw)-2.0/180.0*3.1415926
    // Eigen::Vector3d ea(0.0/180.0*3.1415926, 0.0/180.0*3.1415926, 0.00769);
    // Eigen::Vector3d trans_param(0,0,0);//0.03, 0.15, 0.04

    // //3.1 欧拉角转换为旋转矩阵
    // Eigen::Matrix3d rotation_matrix3;
    // rotation_matrix3 = Eigen::AngleAxisd(ea[0], Eigen::Vector3d::UnitZ()) * 
    //                    Eigen::AngleAxisd(ea[1], Eigen::Vector3d::UnitY()) * 
    //                    Eigen::AngleAxisd(ea[2], Eigen::Vector3d::UnitX());
    // // cout << "rotation matrix3 =\n" << rotation_matrix3 << endl;   

    // //compute new position
    // Eigen::Vector3d new_trans_pos;
    // new_trans_pos = rotation_matrix3 * state.pos_end + trans_param;

    geometry_msgs::Quaternion geoQuat = tf::createQuaternionMsgFromRollPitchYaw
            (euler_cur(0), euler_cur(1), euler_cur(2));
    Eigen::Vector3d twist = state.rot_end.transpose() * state.vel_end;

    nav_msgs::Odometry odomAftMapped;
    odomAftMapped.header.frame_id = map_frame_;//camera_init
    odomAftMapped.child_frame_id = uav_frame_;
    odomAftMapped.header.stamp = ros::Time::now();//ros::Time().fromSec(last_timestamp_lidar);
    odomAftMapped.pose.pose.orientation.x = geoQuat.x;
    odomAftMapped.pose.pose.orientation.y = geoQuat.y;
    odomAftMapped.pose.pose.orientation.z = geoQuat.z;
    odomAftMapped.pose.pose.orientation.w = geoQuat.w;
    odomAftMapped.pose.pose.position.x = state.pos_end(0);
    odomAftMapped.pose.pose.position.y = state.pos_end(1);
    odomAftMapped.pose.pose.position.z = state.pos_end(2);
    odomAftMapped.twist.twist.linear.x = twist(0);
    odomAftMapped.twist.twist.linear.y = twist(1);
    odomAftMapped.twist.twist.linear.z = twist(2);
    pubOdomAftMapped.publish(odomAftMapped);

    static tf::TransformBroadcaster br;
    tf::Transform                   transform;
    tf::Quaternion                  q;
    transform.setOrigin( tf::Vector3( odomAftMapped.pose.pose.position.x,
                                        odomAftMapped.pose.pose.position.y,
                                        odomAftMapped.pose.pose.position.z ) );
    q.setW( odomAftMapped.pose.pose.orientation.w );
    q.setX( odomAftMapped.pose.pose.orientation.x );
    q.setY( odomAftMapped.pose.pose.orientation.y );
    q.setZ( odomAftMapped.pose.pose.orientation.z );
    transform.setRotation( q );
    br.sendTransform( tf::StampedTransform( transform, odomAftMapped.header.stamp, map_frame_, uav_frame_) );

    geometry_msgs::PoseStamped msg_body_pose;
    msg_body_pose.header.stamp = ros::Time::now();
    msg_body_pose.header.frame_id = "/camera_odom_frame";
    msg_body_pose.pose.position.x = state.pos_end(0);
    msg_body_pose.pose.position.y = state.pos_end(1);
    msg_body_pose.pose.position.z = state.pos_end(2);
    msg_body_pose.pose.orientation.x = geoQuat.x;
    msg_body_pose.pose.orientation.y = geoQuat.y;
    msg_body_pose.pose.orientation.z = geoQuat.z;
    msg_body_pose.pose.orientation.w = geoQuat.w;
    #ifdef DEPLOY
    mavros_pose_publisher.publish(msg_body_pose);
    #endif

    // Also publish Mavros vision pose
    geometry_msgs::PoseStamped visionPose;
    visionPose.header.stamp = ros::Time::now();
    visionPose.header.frame_id = map_frame_;
    visionPose.pose.orientation.x = geoQuat.x;
    visionPose.pose.orientation.y = geoQuat.y;
    visionPose.pose.orientation.z = geoQuat.z;
    visionPose.pose.orientation.w = geoQuat.w;
    visionPose.pose.position.x = state.pos_end(0);
    visionPose.pose.position.y = state.pos_end(1);
    visionPose.pose.position.z = state.pos_end(2);
    pubVisionPose.publish(visionPose);

    /******* Publish Path ********/
    msg_body_pose.header.frame_id = map_frame_;

    nav_msgs::Path path;
    path.header.stamp    = ros::Time::now();
    path.header.frame_id = map_frame_;
    path.poses.push_back(msg_body_pose);
    pubPath.publish(path);

    /*** save debug variables ***/
    // frame_num ++;
    // aver_time_consu = aver_time_consu * (frame_num - 1) / frame_num + (t5 - t0) / frame_num;
    // aver_time_consu = aver_time_consu * 0.8 + (t5 - t0) * 0.2;
    // T1.push_back(Measures.lidar_beg_time);
    // s_plot.push_back(aver_time_consu);
    // s_plot2.push_back(t5 - t3);
    // s_plot3.push_back(match_time);
    // s_plot4.push_back(float(feats_down_size/10000.0));
    // s_plot5.push_back(t5 - t0);

    // std::cout<<"[ mapping ]: time: copy map "<<copy_time<<" readd: "<<readd_time<<" match "<<match_time<<" solve "<<solve_time<<"acquire: "<<t4-t3<<" map incre "<<t5-t4<<" total "<<aver_time_consu<<std::endl;
    // fout_out << std::setw(10) << Measures.lidar_beg_time << " " << euler_cur.transpose()*57.3 << " " << state.pos_end.transpose() << " " << state.vel_end.transpose() \
    // <<" "<<state.bias_g.transpose()<<" "<<state.bias_a.transpose()<< std::endl;
    // fout_out<<std::setw(8)<<laserCloudSelNum<<" "<<Measures.lidar_beg_time<<" "<<t2-t0<<" "<<match_time<<" "<<t5-t3<<" "<<t5-t0<<std::endl;
}

void LaserMapping::lasermapFovSegment(std::vector<PointCloudXYZI::Ptr> &featsArray)
{
    laserCloudValidNum = 0;
    bool _last_inFOV[laserCloudNum];
    int laserCloudValidInd[laserCloudNum];

    pointBodyToWorld(XAxisPoint_body, XAxisPoint_world);

    int centerCubeI = int((state.pos_end(0) + 0.5 * cube_len) / cube_len) + laserCloudCenWidth;
    int centerCubeJ = int((state.pos_end(1) + 0.5 * cube_len) / cube_len) + laserCloudCenHeight;
    int centerCubeK = int((state.pos_end(2) + 0.5 * cube_len) / cube_len) + laserCloudCenDepth;

    if (state.pos_end(0) + 0.5 * cube_len < 0) centerCubeI--;
    if (state.pos_end(1) + 0.5 * cube_len < 0) centerCubeJ--;
    if (state.pos_end(2) + 0.5 * cube_len < 0) centerCubeK--;

    bool last_inFOV_flag = 0;
    int  cube_index = 0;
    cub_needrm.clear();
    cub_needad.clear();
    // T2.push_back(Measures.lidar_beg_time);
    double t_begin = omp_get_wtime();

    std::cout<<"centerCubeIJK: "<<centerCubeI<<" "<<centerCubeJ<<" "<<centerCubeK<<std::endl;


    while (centerCubeI < FOV_RANGE + 1)
    {
        for (int j = 0; j < laserCloudHeight; j++)
        {
            for (int k = 0; k < laserCloudDepth; k++)
            {
                int i = laserCloudWidth - 1;

                PointCloudXYZI::Ptr laserCloudCubeSurfPointer = featsArray.at(cubeInd(i, j, k));
                last_inFOV_flag = _last_inFOV[cube_index];

                for (; i >= 1; i--) {
                    featsArray.at(cubeInd(i, j, k))  = featsArray.at(cubeInd(i-1, j, k));
                    _last_inFOV[cubeInd(i, j, k)] = _last_inFOV[cubeInd(i-1, j, k)];
                }

                featsArray.at(cubeInd(i, j, k)) = laserCloudCubeSurfPointer;
                _last_inFOV[cubeInd(i, j, k)] = last_inFOV_flag;
                laserCloudCubeSurfPointer->clear();
            }
        }
        centerCubeI++;
        laserCloudCenWidth++;
    }

    while (centerCubeI >= laserCloudWidth - (FOV_RANGE + 1)) {
        for (int j = 0; j < laserCloudHeight; j++) {
            for (int k = 0; k < laserCloudDepth; k++) {
                int i = 0;

                PointCloudXYZI::Ptr laserCloudCubeSurfPointer = featsArray.at(cubeInd(i, j, k));
                last_inFOV_flag = _last_inFOV[cube_index];

                for (; i >= 1; i--) {
                    featsArray.at(cubeInd(i, j, k))  = featsArray.at(cubeInd(i+1, j, k));
                    _last_inFOV[cubeInd(i, j, k)] = _last_inFOV[cubeInd(i+1, j, k)];
                }

                featsArray.at(cubeInd(i, j, k)) = laserCloudCubeSurfPointer;
                _last_inFOV[cubeInd(i, j, k)] = last_inFOV_flag;
                laserCloudCubeSurfPointer->clear();
            }
        }

        centerCubeI--;
        laserCloudCenWidth--;
    }

    while (centerCubeJ < (FOV_RANGE + 1)) {
        for (int i = 0; i < laserCloudWidth; i++) {
            for (int k = 0; k < laserCloudDepth; k++) {
                int j = laserCloudHeight - 1;

                PointCloudXYZI::Ptr laserCloudCubeSurfPointer = featsArray.at(cubeInd(i, j, k));
                last_inFOV_flag = _last_inFOV[cube_index];

                for (; i >= 1; i--) {
                    featsArray.at(cubeInd(i, j, k))  = featsArray.at(cubeInd(i, j-1, k));
                    _last_inFOV[cubeInd(i, j, k)] = _last_inFOV[cubeInd(i, j-1, k)];
                }

                featsArray.at(cubeInd(i, j, k)) = laserCloudCubeSurfPointer;
                _last_inFOV[cubeInd(i, j, k)] = last_inFOV_flag;
                laserCloudCubeSurfPointer->clear();
            }
        }

        centerCubeJ++;
        laserCloudCenHeight++;
    }

    while (centerCubeJ >= laserCloudHeight - (FOV_RANGE + 1)) {
        for (int i = 0; i < laserCloudWidth; i++) {
            for (int k = 0; k < laserCloudDepth; k++) {
                int j = 0;
                PointCloudXYZI::Ptr laserCloudCubeSurfPointer = featsArray.at(cubeInd(i, j, k));
                last_inFOV_flag = _last_inFOV[cube_index];

                for (; i >= 1; i--) {
                    featsArray.at(cubeInd(i, j, k))  = featsArray.at(cubeInd(i, j+1, k));
                    _last_inFOV[cubeInd(i, j, k)] = _last_inFOV[cubeInd(i, j+1, k)];
                }

                featsArray.at(cubeInd(i, j, k)) = laserCloudCubeSurfPointer;
                _last_inFOV[cubeInd(i, j, k)] = last_inFOV_flag;
                laserCloudCubeSurfPointer->clear();
            }
        }

        centerCubeJ--;
        laserCloudCenHeight--;
    }

    while (centerCubeK < (FOV_RANGE + 1)) {
        for (int i = 0; i < laserCloudWidth; i++) {
            for (int j = 0; j < laserCloudHeight; j++) {
                int k = laserCloudDepth - 1;
                PointCloudXYZI::Ptr laserCloudCubeSurfPointer = featsArray.at(cubeInd(i, j, k));
                last_inFOV_flag = _last_inFOV[cube_index];

                for (; i >= 1; i--) {
                    featsArray.at(cubeInd(i, j, k))  = featsArray.at(cubeInd(i, j, k-1));
                    _last_inFOV[cubeInd(i, j, k)] = _last_inFOV[cubeInd(i, j, k-1)];
                }

                featsArray.at(cubeInd(i, j, k)) = laserCloudCubeSurfPointer;
                _last_inFOV[cubeInd(i, j, k)] = last_inFOV_flag;
                laserCloudCubeSurfPointer->clear();
            }
        }

        centerCubeK++;
        laserCloudCenDepth++;
    }

    while (centerCubeK >= laserCloudDepth - (FOV_RANGE + 1))
    {
        for (int i = 0; i < laserCloudWidth; i++)
        {
            for (int j = 0; j < laserCloudHeight; j++)
            {
                int k = 0;
                PointCloudXYZI::Ptr laserCloudCubeSurfPointer = featsArray.at(cubeInd(i, j, k));
                last_inFOV_flag = _last_inFOV[cube_index];

                for (; i >= 1; i--) {
                    featsArray.at(cubeInd(i, j, k))  = featsArray.at(cubeInd(i, j, k+1));
                    _last_inFOV[cubeInd(i, j, k)] = _last_inFOV[cubeInd(i, j, k+1)];
                }

                featsArray.at(cubeInd(i, j, k)) = laserCloudCubeSurfPointer;
                _last_inFOV[cubeInd(i, j, k)] = last_inFOV_flag;
                laserCloudCubeSurfPointer->clear();
            }
        }

        centerCubeK--;
        laserCloudCenDepth--;
    }

    cube_points_add->clear();
    featsFromMap->clear();
    bool now_inFOV[laserCloudNum] = {false};

    // std::cout<<"centerCubeIJK: "<<centerCubeI<<" "<<centerCubeJ<<" "<<centerCubeK<<std::endl;
    // std::cout<<"laserCloudCen: "<<laserCloudCenWidth<<" "<<laserCloudCenHeight<<" "<<laserCloudCenDepth<<std::endl;

    for (int i = centerCubeI - FOV_RANGE; i <= centerCubeI + FOV_RANGE; i++) 
    {
        for (int j = centerCubeJ - FOV_RANGE; j <= centerCubeJ + FOV_RANGE; j++) 
        {
            for (int k = centerCubeK - FOV_RANGE; k <= centerCubeK + FOV_RANGE; k++) 
            {
                if (i >= 0 && i < laserCloudWidth &&
                        j >= 0 && j < laserCloudHeight &&
                        k >= 0 && k < laserCloudDepth) 
                {
                    Eigen::Vector3f center_p(cube_len * (i - laserCloudCenWidth), \
                                             cube_len * (j - laserCloudCenHeight), \
                                             cube_len * (k - laserCloudCenDepth));

                    float check1, check2;
                    float squaredSide1, squaredSide2;
                    float ang_cos = 1;
                    bool &last_inFOV = _last_inFOV[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k];
                    bool inFOV = centerinFOV(center_p);

                    for (int ii = -1; (ii <= 1) && (!inFOV); ii += 2) 
                    {
                        for (int jj = -1; (jj <= 1) && (!inFOV); jj += 2) 
                        {
                            for (int kk = -1; (kk <= 1) && (!inFOV); kk += 2) 
                            {
                                Eigen::Vector3f corner_p(cube_len * ii, cube_len * jj, cube_len * kk);
                                corner_p = center_p + 0.5 * corner_p;
                                
                                inFOV = cornerinFOV(corner_p);
                            }
                        }
                    }

                    now_inFOV[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] = inFOV;

                    /*** readd cubes and points ***/
                    if (inFOV)
                    {
                        int center_index = i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k;
                        *cube_points_add += *featsArray.at(center_index);
                        featsArray.at(center_index)->clear();
                        if (!last_inFOV)
                        {
                            BoxPointType cub_points;
                            for(int i = 0; i < 3; i++)
                            {
                                cub_points.vertex_max[i] = center_p[i] + 0.5 * cube_len;
                                cub_points.vertex_min[i] = center_p[i] - 0.5 * cube_len;
                            }
                            cub_needad.push_back(cub_points);
                            laserCloudValidInd[laserCloudValidNum] = center_index;
                            laserCloudValidNum ++;
                            // std::cout<<"readd center: "<<center_p.transpose()<<std::endl;
                        }
                    }
                }
            }
        }
    }

    for (int i = 0; i < laserCloudWidth; i++) 
    {
        for (int j = 0; j < laserCloudHeight; j++) 
        {
            for (int k = 0; k < laserCloudDepth; k++) 
            {
                int ind = i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k;
                if((!now_inFOV[ind]) && _last_inFOV[ind])
                {
                    BoxPointType cub_points;
                    Eigen::Vector3f center_p(cube_len * (i - laserCloudCenWidth),\
                                             cube_len * (j - laserCloudCenHeight),\
                                             cube_len * (k - laserCloudCenDepth));
                    // std::cout<<"center_p: "<<center_p.transpose()<<std::endl;

                    for(int i = 0; i < 3; i++)
                    {
                        cub_points.vertex_max[i] = center_p[i] + 0.5 * cube_len;
                        cub_points.vertex_min[i] = center_p[i] - 0.5 * cube_len;
                    }
                    cub_needrm.push_back(cub_points);
                }
                _last_inFOV[ind] = now_inFOV[ind];
            }
        }
    }

    copy_time = omp_get_wtime() - t_begin;

    if(cub_needrm.size() > 0)               ikdtree.Delete_Point_Boxes(cub_needrm);
    // s_plot4.push_back(omp_get_wtime() - t_begin); t_begin = omp_get_wtime();
    if(cub_needad.size() > 0)               ikdtree.Add_Point_Boxes(cub_needad); 
    // s_plot5.push_back(omp_get_wtime() - t_begin); t_begin = omp_get_wtime();
    if(cube_points_add->points.size() > 0)  ikdtree.Add_Points(cube_points_add->points, true);
}

int LaserMapping::cubeInd(const int &i, const int &j, const int &k) {
    return (i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k);
}

bool LaserMapping::centerinFOV(Eigen::Vector3f cube_p) {                                
    Eigen::Vector3f dis_vec = state.pos_end.cast<float>() - cube_p;
    float squaredSide1 = dis_vec.transpose() * dis_vec;

    if(squaredSide1 < 0.4 * cube_len * cube_len) return true;

    dis_vec = XAxisPoint_world.cast<float>() - cube_p;
    float squaredSide2 = dis_vec.transpose() * dis_vec;

    float ang_cos = fabs(squaredSide1 <= 3) ? 1.0 :
        (LIDAR_SP_LEN * LIDAR_SP_LEN + squaredSide1 - squaredSide2) / (2 * LIDAR_SP_LEN * sqrt(squaredSide1));
    
    return ((ang_cos > HALF_FOV_COS)? true : false);
}

bool LaserMapping::cornerinFOV(Eigen::Vector3f cube_p) {                                
    Eigen::Vector3f dis_vec = state.pos_end.cast<float>() - cube_p;
    float squaredSide1 = dis_vec.transpose() * dis_vec;

    dis_vec = XAxisPoint_world.cast<float>() - cube_p;
    float squaredSide2 = dis_vec.transpose() * dis_vec;

    float ang_cos = fabs(squaredSide1 <= 3) ? 1.0 :
        (LIDAR_SP_LEN * LIDAR_SP_LEN + squaredSide1 - squaredSide2) / (2 * LIDAR_SP_LEN * sqrt(squaredSide1));
    
    return ((ang_cos > HALF_FOV_COS)? true : false);
}

//project lidar frame to world
void LaserMapping::pointBodyToWorld(PointType const * const pi, PointType * const po)
{
    Eigen::Vector3d p_body(pi->x, pi->y, pi->z);
    Eigen::Vector3d p_global(state.rot_end * (p_body + Lidar_offset_to_IMU) + state.pos_end);
    
    po->x = p_global(0);
    po->y = p_global(1);
    po->z = p_global(2);
    po->intensity = pi->intensity;
}

template<typename T>
void LaserMapping::pointBodyToWorld(const Eigen::Matrix<T, 3, 1> &pi, Eigen::Matrix<T, 3, 1> &po)
{
    Eigen::Vector3d p_body(pi[0], pi[1], pi[2]);
    Eigen::Vector3d p_global(state.rot_end * (p_body + Lidar_offset_to_IMU) + state.pos_end);
    po[0] = p_global(0);
    po[1] = p_global(1);
    po[2] = p_global(2);
}

void LaserMapping::rgbPointBodyToWorld(PointType const * const pi, pcl::PointXYZI * const po)
{
    Eigen::Vector3d p_body(pi->x, pi->y, pi->z);
    Eigen::Vector3d p_global(state.rot_end * (p_body + Lidar_offset_to_IMU) + state.pos_end);
    
    po->x = p_global(0);
    po->y = p_global(1);
    po->z = p_global(2);
    po->intensity = pi->intensity;

    float intensity = pi->intensity;
    intensity = intensity - std::floor(intensity);

    int reflection_map = intensity*10000;
}

}