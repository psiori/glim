#include <glim_cloud_fusion/fusion.hpp>

#include <algorithm>
#include <cmath>

#include "fusion_detail.hpp"

namespace glim_cloud_fusion {

bool normalize_lidar_scan_times(
  const LidarScan& scan,
  const ScanTimeParams& params,
  std::vector<double>& out_times,
  double& out_duration,
  std::string& err) {
  if (scan.times.empty()) {
    err = "scan has no per-point times";
    return false;
  }
  const double scale = (params.time_unit == TimeUnit::Nanoseconds) ? 1e-9 : 1.0;
  out_times.resize(scan.times.size());
  for (size_t i = 0; i < scan.times.size(); ++i) {
    out_times[i] = scan.times[i] * scale;
  }

  if (!params.scan_duration.has_value() || *params.scan_duration <= 0.0 || !std::isfinite(*params.scan_duration)) {
    err = "scan_duration is required";
    return false;
  }
  out_duration = *params.scan_duration;

  if (params.time_origin == TimeOrigin::FirstPoint) {
    const double t0 = *std::min_element(out_times.begin(), out_times.end());
    for (auto& t : out_times) {
      t -= t0;
    }
  }
  return true;
}

FusionResult fuse_scans(const std::vector<LidarScan>& scans, const FusionParams& params, FusionCallback on_progress) {
  FusionResult result;
  if (scans.empty()) {
    result.error = FusionError::EmptyInput;
    result.message = "no scans";
    return result;
  }

  if (scans.size() == 1) {
    std::string err;
    const auto frame = detail::to_preprocessed(scans.front(), params, err);
    if (!frame) {
      result.error = FusionError::InvalidArgument;
      result.message = err;
      return result;
    }
    std::vector<Eigen::Vector4d, Eigen::aligned_allocator<Eigen::Vector4d>> pts;
    std::vector<double> intensities;
    pts.reserve(frame->points.size());
    intensities.reserve(frame->intensities.size());
    for (size_t i = 0; i < frame->points.size(); ++i) {
      pts.push_back(frame->points[i]);
      intensities.push_back(frame->intensities.empty() ? 0.0 : frame->intensities[i]);
    }
    result.cloud = detail::voxel_merge(pts, intensities, params.voxel_resolution);
    result.scan_poses.push_back(Eigen::Isometry3d::Identity());
    if (on_progress) {
      FusionProgress progress;
      progress.scan_index = 1;
      progress.total_scans = 1;
      progress.stage = "passthrough";
      progress.preview_cloud = result.cloud;
      on_progress(progress);
    }
    return result;
  }

  if (params.backend == FusionBackend::CtIntegrator) {
    return detail::fuse_scans_ct_integrator(scans, params, on_progress);
  }

  return detail::fuse_scans_slam_mapping(scans, params, on_progress);
}

}  // namespace glim_cloud_fusion
