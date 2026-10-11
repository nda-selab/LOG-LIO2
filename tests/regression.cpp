// Exercise the actual mapping helpers without starting ROS subscriptions.
#define main log_lio_node_main
#include "../src/voxelMapping.cpp"
#undef main
#include <stdexcept>

void require(bool condition, const char *message)
{
    if (!condition) throw std::runtime_error(message);
}

sensor_msgs::PointCloud2::Ptr cloudMessage(const std::vector<float> &times)
{
    pcl::PointCloud<PointXYZIRT> cloud;
    for (size_t i = 0; i < times.size(); ++i) {
        PointXYZIRT p{};
        p.x = 5 * std::cos(-0.1 * i);
        p.y = 5 * std::sin(-0.1 * i);
        p.z = 1;
        p.ring = 0;
        p.time = times[i];
        cloud.push_back(p);
    }
    sensor_msgs::PointCloud2::Ptr msg(new sensor_msgs::PointCloud2);
    pcl::toROSMsg(cloud, *msg);
    msg->header.stamp = ros::Time(100);
    return msg;
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "log_lio_regression",
              ros::init_options::NoSigintHandler | ros::init_options::NoRosout);
    ros::Time::init();
    p_pre->lidar_type = VELO16;
    p_pre->N_SCANS = 16;
    p_pre->SCAN_RATE = 10;
    p_pre->compute_normal = false;
    PointCloudXYZI::Ptr output(new PointCloudXYZI);
    const float units[] = {1, 1000, 1000000, 1000000000};
    for (int unit = SEC; unit <= NS; ++unit) {
        p_pre->time_unit = unit;
        auto msg = cloudMessage({0, 0.05f * units[unit], 0.10f * units[unit],
                                 0.2f * units[unit], -0.01f * units[unit],
                                 std::numeric_limits<float>::quiet_NaN(), 0});
        p_pre->process(msg, output);
        require(output->size() == 4, "timestamp filtering must agree across units");
        require(std::abs(output->points[1].curvature - 50) < 0.001,
                "50 ms point must retain its offset");
        require(std::abs(output->points[2].curvature - 100) < 0.001,
                "100 ms point must retain its offset");
    }
    p_pre->time_unit = SEC;
    p_pre->process(cloudMessage({0, 0, 0}), output);
    require(output->size() == 2 && output->back().curvature > 0,
            "missing timestamps must use yaw-based offsets");

    PointCloudXYZI::Ptr input(new PointCloudXYZI);
    PointType p{};
    p.x = 2; p.y = 3; p.z = 4; p.normal_z = 1;
    input->push_back(p);
    cov_measurement.assign(1, M3D::Identity() * 0.01);
    state_ikfom state = kf.get_x();
    state.offset_T_L_I = V3D(0.1, 0.2, 0.3);
    state.rot = SO3(Exp(V3D(0.1, -0.2, 0.3)));
    kf.change_x(state);
    std::vector<pointWithCov> before, after;
    prepareMapping(state, input, before);
    state.pos += V3D(100, -200, 50);
    kf.change_x(state);
    prepareMapping(state, input, after);
    require((before[0].cov - after[0].cov).norm() < 1e-10,
            "global translation must not change point covariance");
    require((after[0].point - before[0].point - state.pos).norm() < 1e-10,
            "global translation must change map coordinates");

    scan_count = 719999;
    auto msg = cloudMessage({0, 0.05f, 0.1f});
    for (int i = 0; i < 3; ++i) standard_pcl_cbk(msg);
    require(scan_count == 720002 && std::isfinite(mean_preprocess),
            "timing statistics must work beyond the former array limit");
    lidar_pushed = true;
    imu_buffer.push_back(sensor_msgs::Imu::Ptr(new sensor_msgs::Imu));
    msg->header.stamp = ros::Time(99);
    standard_pcl_cbk(msg);
    require(lidar_buffer.empty() && time_buffer.empty() && imu_buffer.empty() && !lidar_pushed,
            "LiDAR rewind must clear all synchronized buffers");
    lidar_buffer.push_back(output);
    time_buffer.push_back(100);
    auto imu = boost::make_shared<sensor_msgs::Imu>();
    imu->header.stamp = ros::Time(99);
    imu_buffer.push_back(imu);
    last_timestamp_imu = 100;
    imu_cbk(imu);
    require(lidar_buffer.empty() && time_buffer.empty() && imu_buffer.empty(),
            "IMU rewind must clear all synchronized buffers");
    saveTraj("/tmp/loglio2-empty-trajectory-test.txt");
    saveVoxelMap("/tmp/loglio2-empty-map-test.pcd");
    std::cout << "Regression checks passed" << std::endl;
}
