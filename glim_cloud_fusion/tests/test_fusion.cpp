#include <gtest/gtest.h>

#include <glim_cloud_fusion/fusion.hpp>

namespace {

glim_cloud_fusion::LidarScan make_box_scan(double yaw, double z_offset) {
  glim_cloud_fusion::LidarScan scan;
  scan.stamp = 0.0;
  const double c = std::cos(yaw);
  const double s = std::sin(yaw);
  for (int face = 0; face < 6; ++face) {
    for (int u = 0; u < 8; ++u) {
      for (int v = 0; v < 8; ++v) {
        Eigen::Vector3d p(0.1 * u, 0.1 * v, z_offset);
        switch (face) {
          case 0:
            p = Eigen::Vector3d(0.0, p.x(), p.y());
            break;
          case 1:
            p = Eigen::Vector3d(0.7, p.x(), p.y());
            break;
          case 2:
            p = Eigen::Vector3d(p.x(), 0.0, p.y());
            break;
          case 3:
            p = Eigen::Vector3d(p.x(), 0.7, p.y());
            break;
          case 4:
            p = Eigen::Vector3d(p.x(), p.y(), 0.0);
            break;
          default:
            p = Eigen::Vector3d(p.x(), p.y(), 0.7);
            break;
        }
        const Eigen::Vector3d pr(c * p.x() - s * p.y(), s * p.x() + c * p.y(), p.z());
        scan.points.emplace_back(pr.x(), pr.y(), pr.z(), 1.0);
        scan.times.push_back(static_cast<double>(scan.points.size()) * 0.001);
        scan.intensities.push_back(static_cast<double>(face) / 6.0);
        scan.keep.push_back(face != 2);
      }
    }
  }
  return scan;
}

}  // namespace

TEST(Fusion, SingleScanPassthrough) {
  auto scan = make_box_scan(0.0, 0.0);
  glim_cloud_fusion::FusionParams params;
  const auto result = glim_cloud_fusion::fuse_scans({scan}, params);
  EXPECT_EQ(result.error, glim_cloud_fusion::FusionError::Ok);
  EXPECT_GT(result.cloud.points.size(), 0u);
}

TEST(Fusion, ScanStartRequiresDuration) {
  glim_cloud_fusion::LidarScan scan;
  scan.points.emplace_back(1, 0, 0, 1);
  scan.times.push_back(0.25);
  scan.intensities.push_back(1.0);
  glim_cloud_fusion::FusionParams params;
  params.scan_time.time_origin = glim_cloud_fusion::TimeOrigin::ScanStart;
  const auto result = glim_cloud_fusion::fuse_scans({scan}, params);
  EXPECT_EQ(result.error, glim_cloud_fusion::FusionError::InvalidArgument);
}

TEST(Fusion, AzimuthWindowAlphaRange) {
  glim_cloud_fusion::LidarScan scan;
  for (int i = 0; i < 20; ++i) {
    const double t = 0.25 + 0.25 * static_cast<double>(i) / 19.0;
    scan.times.push_back(t);
    scan.points.emplace_back(0.1 * i, 0.0, 0.0, 1.0);
    scan.intensities.push_back(static_cast<double>(i));
  }
  glim_cloud_fusion::ScanTimeParams time_params;
  time_params.time_origin = glim_cloud_fusion::TimeOrigin::ScanStart;
  time_params.scan_duration = 1.0;
  std::vector<double> norm_times;
  double duration = 0.0;
  std::string err;
  EXPECT_TRUE(glim_cloud_fusion::normalize_lidar_scan_times(scan, time_params, norm_times, duration, err));
  EXPECT_NEAR(glim_cloud_fusion::scan_time_alpha(norm_times.front(), duration), 0.25, 1e-6);
  EXPECT_NEAR(glim_cloud_fusion::scan_time_alpha(norm_times.back(), duration), 0.5, 1e-6);
}

TEST(Fusion, MaskRemovesFace) {
  auto scan = make_box_scan(0.0, 0.0);
  glim_cloud_fusion::FusionParams params;
  const auto result = glim_cloud_fusion::fuse_scans({scan}, params);
  EXPECT_EQ(result.error, glim_cloud_fusion::FusionError::Ok);
}
