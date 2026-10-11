// Exercise the actual mapping helpers without starting ROS subscriptions.
// Keep the node's optional debug output outside the source tree.
#undef ROOT_DIR
#define ROOT_DIR "/tmp/loglio2-regression/"
#define main log_lio_node_main
#include "../src/voxelMapping.cpp"
#undef main
#include <stdexcept>
#include <set>

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

sensor_msgs::Imu::Ptr imuSample(double time, double yaw_rate = 0)
{
    auto imu = boost::make_shared<sensor_msgs::Imu>();
    imu->header.stamp.fromSec(time);
    imu->linear_acceleration.z = G_m_s2;
    imu->angular_velocity.z = yaw_rate;
    return imu;
}

void testImuProcessing()
{
    ImuProcess processor;
    esekfom::esekf<state_ikfom, 12, input_ikfom> filter;
    double limits[23];
    std::fill(limits, limits + 23, 0.001);
    filter.init_dyn_share(get_f, df_dx, df_dw, observation_model_share, 3, limits);
    const M3D extrinsic = Exp(V3D(0.3, 0.2, -0.1));
    processor.set_extrinsic(V3D::Zero(), extrinsic);
    PointCloudXYZI::Ptr output(new PointCloudXYZI);
    PointType point{};
    point.x = 1;
    point.normal_x = 1;

    // Repeat after Reset with a new time epoch to exercise reinitialization.
    for (double start : {100.0, 200.0}) {
        processor.Reset();
        state_ikfom initial;
        filter.change_x(initial);
        MeasureGroup initialization;
        initialization.lidar_beg_time = start;
        initialization.lidar_end_time = start + 0.1;
        initialization.lidar->push_back(point);
        for (int i = 0; i <= 20; ++i)
            initialization.imu.push_back(imuSample(start + i * 0.0045));
        output->push_back(point);
        processor.Process(initialization, filter, output);
        require(output->empty(), "initialization must not return stale points");
        // Account for the filter's fixed S2 gravity magnitude (9.809), while
        // the accelerometer normalization uses G_m_s2 (9.81).
        auto calibrated = filter.get_x();
        calibrated.ba.z() = G_m_s2 - calibrated.grav.get_vect().norm();
        filter.change_x(calibrated);

        MeasureGroup scan;
        scan.lidar_beg_time = start + 0.1;
        scan.lidar_end_time = start + 0.2;
        for (float offset : {0.f, 50.f, 100.f}) {
            point.curvature = offset;
            scan.lidar->push_back(point);
        }
        scan.imu.push_back(imuSample(start + 0.15, 1));
        scan.imu.push_back(imuSample(start + 0.2, 1));
        processor.Process(scan, filter, output);
        require(output->size() == 3 && filter.get_P().allFinite(),
                "first scan after initialization must remain finite");
        const auto state = filter.get_x();
        require((state.rot.toRotationMatrix() - Exp(V3D(0, 0, 0.075))).norm() < 1e-8,
                "integration must start at initialization scan end");
        require(state.pos.norm() < 1e-8 && state.vel.norm() < 1e-8,
                "stationary acceleration must cancel gravity");
        const double angles[] = {-0.075, -0.05, 0};
        for (size_t i = 0; i < output->size(); ++i) {
            const V3D expected = extrinsic.transpose() * Exp(V3D(0, 0, angles[i])) *
                                 extrinsic * V3D::UnitX();
            const V3D normal = output->points[i].getNormalVector3fMap().cast<double>();
            require((normal - expected).norm() < 1e-6,
                    "normal must be rotated into the scan-end LiDAR frame");
            require((output->points[i].getVector3fMap().cast<double>() - expected).norm() < 1e-6,
                    "point and normal rotations must agree, including zero offset");
        }
        auto missing_imu = scan;
        missing_imu.imu.clear();
        processor.Process(missing_imu, filter, output);
        require(output->empty(), "missing IMU must clear previous output");
        output->push_back(point);
        auto empty_scan = scan;
        empty_scan.lidar.reset(new PointCloudXYZI);
        processor.Process(empty_scan, filter, output);
        require(output->empty(), "empty scan must be rejected before deskew");
        output->push_back(point);
        empty_scan.lidar.reset();
        processor.Process(empty_scan, filter, output);
        require(output->empty(), "missing scan must clear previous output");
    }
}

void testMissingImuSynchronization()
{
    lidar_buffer.clear();
    time_buffer.clear();
    imu_buffer.clear();
    lidar_pushed = false;
    PointCloudXYZI::Ptr scan(new PointCloudXYZI(2, 1));
    scan->back().curvature = 100;
    lidar_buffer.push_back(scan);
    time_buffer.push_back(10);
    imu_buffer.push_back(imuSample(10.2));
    last_timestamp_imu = 10.2;
    MeasureGroup measure;
    require(!sync_packages(measure), "scan without IMU must not synchronize");
    require(lidar_buffer.empty() && time_buffer.empty() && imu_buffer.size() == 1,
            "drop unmatched scan but preserve future IMU");
    lidar_buffer.push_back(scan);
    time_buffer.push_back(10.1);
    require(sync_packages(measure) && measure.imu.size() == 1,
            "IMU at exact scan end must be included in the following scan");
    require(imu_buffer.empty(), "matched IMU must be consumed once");
}

void testParallelPlaneIds()
{
    std::vector<pointWithCov> points;
    for (int x = -1; x <= 1; ++x) {
        for (int y = -1; y <= 1; ++y) {
            pointWithCov p{};
            p.point = V3D(x, y * 2, 0);
            p.cov = M3D::Identity() * 0.001;
            points.push_back(p);
        }
    }
    std::vector<int> ids(256);
#pragma omp parallel for num_threads(4)
    for (int i = 0; i < static_cast<int>(ids.size()); ++i) {
        // Exercise both plane and non-plane ID allocation paths.
        OctoTree tree(0, 0, {5}, 100, 100, i % 2 ? 0.01f : -1.0f);
        tree.init_plane(points, tree.plane_ptr_);
        ids[i] = tree.plane_ptr_->id;
        delete tree.plane_ptr_;
    }
    require(std::set<int>(ids.begin(), ids.end()).size() == ids.size(),
            "concurrently initialized planes must have unique IDs");
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

    testImuProcessing();
    testMissingImuSynchronization();
    testParallelPlaneIds();

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
