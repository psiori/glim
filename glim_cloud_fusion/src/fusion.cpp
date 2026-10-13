#include <glim_cloud_fusion/fusion.hpp>

#include <algorithm>
#include <cmath>

#include <glim/odometry/callbacks.hpp>
#include <glim/odometry/odometry_estimation_ct.hpp>

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

  const glim::OdometryEstimationCTParams odom_params = detail::odometry_params_from_fusion(params);
  glim::OdometryEstimationCT odom(odom_params);

  struct Snapshot {
    std::vector<Eigen::Vector4d, Eigen::aligned_allocator<Eigen::Vector4d>> points;
    std::vector<double> intensities;
    Eigen::Isometry3d T_world_begin = Eigen::Isometry3d::Identity();
  };
  std::vector<Snapshot> snapshots;

  const auto preview_from_snapshots = [&](size_t count) -> DensePointCloud {
    std::vector<Eigen::Vector4d, Eigen::aligned_allocator<Eigen::Vector4d>> merged_pts;
    std::vector<double> merged_intensities;
    const size_t n = std::min(count, snapshots.size());
    for (size_t si = 0; si < n; ++si) {
      const auto& snap = snapshots[si];
      const Eigen::Isometry3d T = snap.T_world_begin;
      for (size_t i = 0; i < snap.points.size(); ++i) {
        merged_pts.push_back(T * snap.points[i]);
        merged_intensities.push_back(snap.intensities[i]);
      }
    }
    return detail::voxel_merge(merged_pts, merged_intensities, params.voxel_resolution);
  };

  const size_t preview_stride = std::max<size_t>(1, scans.size() / 10);

  for (size_t si = 0; si < scans.size(); ++si) {
    std::string err;
    const auto preprocessed = detail::to_preprocessed(scans[si], params, err);
    if (!preprocessed) {
      result.error = FusionError::InvalidArgument;
      result.message = err;
      return result;
    }

    std::vector<glim::EstimationFrame::ConstPtr> marginalized;
    const auto odom_frame = odom.insert_frame(preprocessed, marginalized);

    Snapshot snap;
    if (odom_frame && odom_frame->frame) {
      snap.T_world_begin = odom_frame->T_world_sensor();
      for (int i = 0; i < odom_frame->frame->size(); ++i) {
        snap.points.push_back(odom_frame->frame->points[i]);
        double intensity = 0.0;
        if (preprocessed->intensities.size() == static_cast<size_t>(odom_frame->frame->size())) {
          intensity = preprocessed->intensities[i];
        }
        snap.intensities.push_back(intensity);
      }
    }
    snapshots.push_back(std::move(snap));

    if (on_progress) {
      FusionProgress progress;
      progress.scan_index = static_cast<int>(si + 1);
      progress.total_scans = static_cast<int>(scans.size());
      progress.stage = "odometry";
      const bool emit_preview =
        si == 0 || si + 1 == scans.size() || (si + 1) % preview_stride == 0;
      if (emit_preview) {
        progress.preview_cloud = preview_from_snapshots(si + 1);
      }
      on_progress(progress);
    }
  }

  std::vector<Eigen::Vector4d, Eigen::aligned_allocator<Eigen::Vector4d>> merged_pts;
  std::vector<double> merged_intensities;
  for (const auto& snap : snapshots) {
    const Eigen::Isometry3d T = snap.T_world_begin;
    for (size_t i = 0; i < snap.points.size(); ++i) {
      merged_pts.push_back(T * snap.points[i]);
      merged_intensities.push_back(snap.intensities[i]);
    }
  }

  result.cloud = detail::voxel_merge(merged_pts, merged_intensities, params.voxel_resolution);
  result.error = FusionError::Ok;
  return result;
}

}  // namespace glim_cloud_fusion
