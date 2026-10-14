#include "fusion_detail.hpp"
#include "fusion_parallel.hpp"

#include <algorithm>
#include <cmath>
#include <random>
#include <unordered_map>
#include <vector>

#include <gtsam/geometry/Pose3.h>

#include <glim/mapping/global_mapping.hpp>
#include <glim/mapping/sub_mapping.hpp>
#include <glim/odometry/odometry_estimation_ct.hpp>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace glim_cloud_fusion::detail {

namespace {

struct OutputVoxelKey {
  int x, y, z;
  bool operator==(const OutputVoxelKey& o) const { return x == o.x && y == o.y && z == o.z; }
};

struct OutputVoxelKeyHash {
  size_t operator()(const OutputVoxelKey& k) const {
    return (static_cast<size_t>(k.x) * 73856093u) ^ (static_cast<size_t>(k.y) * 19349663u) ^ (static_cast<size_t>(k.z) * 83492791u);
  }
};

using OutputVoxelMap = std::unordered_map<OutputVoxelKey, Eigen::Vector4d, OutputVoxelKeyHash>;

std::vector<int> sort_indices_by_time(const std::vector<double>& times) {
  std::vector<int> order(times.size());
  for (size_t i = 0; i < times.size(); ++i) {
    order[i] = static_cast<int>(i);
  }
  std::sort(order.begin(), order.end(), [&](int a, int b) { return times[a] < times[b]; });
  return order;
}

void accumulate_deskewed_scan(
  const glim::PreprocessedFrame& raw,
  const Eigen::Isometry3d& T_world_begin,
  const Eigen::Isometry3d& T_world_end,
  double scan_duration,
  double voxel_resolution,
  double min_distance,
  OutputVoxelMap& voxels,
  int num_threads) {
  const std::vector<int> time_order = sort_indices_by_time(raw.times);
  const double inv_dur = 1.0 / std::max(1e-9, scan_duration);
  const gtsam::Pose3 begin(T_world_begin.matrix());
  const gtsam::Pose3 end(T_world_end.matrix());
  const int n = static_cast<int>(time_order.size());
  if (n == 0) {
    return;
  }

  const int nt = std::max(1, fusion_effective_num_threads(num_threads));
#ifdef _OPENMP
  std::vector<OutputVoxelMap> local_maps(static_cast<size_t>(nt));
#pragma omp parallel num_threads(nt)
  {
    OutputVoxelMap& local = local_maps[omp_get_thread_num()];
#pragma omp for schedule(static)
    for (int ii = 0; ii < n; ++ii) {
      const int idx = time_order[ii];
      if (raw.points[idx].head<3>().norm() < min_distance) {
        continue;
      }
      const double alpha = raw.times[idx] * inv_dur;
      const gtsam::Pose3 T = begin.interpolateRt(end, alpha);
      const Eigen::Vector4d pt = T.matrix() * raw.points[idx];
      const OutputVoxelKey key{
        static_cast<int>(std::floor(pt.x() / voxel_resolution)),
        static_cast<int>(std::floor(pt.y() / voxel_resolution)),
        static_cast<int>(std::floor(pt.z() / voxel_resolution))};
      const double intensity = raw.intensities.empty() ? 0.0 : raw.intensities[idx];
      local[key] = Eigen::Vector4d(pt.x(), pt.y(), pt.z(), intensity);
    }
  }
  for (auto& local : local_maps) {
    for (auto& [key, value] : local) {
      voxels[key] = value;
    }
  }
#else
  for (int ii = 0; ii < n; ++ii) {
    const int idx = time_order[ii];
    if (raw.points[idx].head<3>().norm() < min_distance) {
      continue;
    }
    const double alpha = raw.times[idx] * inv_dur;
    const gtsam::Pose3 T = begin.interpolateRt(end, alpha);
    const Eigen::Vector4d pt = T.matrix() * raw.points[idx];
    const OutputVoxelKey key{
      static_cast<int>(std::floor(pt.x() / voxel_resolution)),
      static_cast<int>(std::floor(pt.y() / voxel_resolution)),
      static_cast<int>(std::floor(pt.z() / voxel_resolution))};
    const double intensity = raw.intensities.empty() ? 0.0 : raw.intensities[idx];
    voxels[key] = Eigen::Vector4d(pt.x(), pt.y(), pt.z(), intensity);
  }
#endif
}

DensePointCloud dense_from_voxels(const OutputVoxelMap& voxels) {
  DensePointCloud out;
  out.points.reserve(voxels.size());
  out.intensities.reserve(voxels.size());
  for (const auto& [_, v] : voxels) {
    out.points.emplace_back(v.x(), v.y(), v.z());
    out.intensities.push_back(v.w());
  }
  return out;
}

int slam_odometry_target_points(const FusionParams& params) {
  const int from_slam = params.slam_mapping.odometry_target_num_points;
  if (from_slam > 0) {
    return from_slam;
  }
  return params.target_num_points;
}

}  // namespace

glim::SubMappingParams sub_mapping_params_from_fusion(const FusionParams& params) {
  glim::SubMappingParams sub(glim::SubMappingParams::FromCaller{});
  const auto& sm = params.slam_mapping;
  if (!sm.enable_sub_mapping) {
    sub.enable_optimization = false;
    return sub;
  }
  sub.max_num_keyframes = sm.submap_max_num_keyframes;
  sub.keyframe_update_strategy = sm.keyframe_update_strategy;
  sub.max_keyframe_overlap = sm.max_keyframe_overlap;
  sub.keyframe_update_interval_trans = sm.keyframe_update_interval_trans;
  sub.keyframe_update_interval_rot = sm.keyframe_update_interval_rot;
  sub.submap_downsample_resolution = sm.submap_downsample_resolution;
  const int target = slam_odometry_target_points(params);
  sub.keyframe_randomsampling_rate = std::min(1.0, static_cast<double>(target) / 60000.0);
  sub.keyframe_randomsampling_rate = std::max(sub.keyframe_randomsampling_rate, 0.05);
  return sub;
}

glim::GlobalMappingParams global_mapping_params_from_fusion(const FusionParams& params) {
  glim::GlobalMappingParams global(glim::GlobalMappingParams::FromCaller{});
  const auto& sm = params.slam_mapping;
  global.enable_optimization = sm.enable_global_mapping;
  global.min_implicit_loop_overlap = sm.global_min_implicit_loop_overlap;
  global.max_implicit_loop_distance = sm.global_max_implicit_loop_distance;
  return global;
}

FusionResult fuse_scans_slam_mapping(
  const std::vector<LidarScan>& scans,
  const FusionParams& params,
  FusionCallback on_progress) {
  FusionResult result;

  glim::OdometryEstimationCTParams odom_params = odometry_params_from_fusion(params);
  if (params.slam_mapping.enable_odometry_smoother) {
    odom_params.enable_fixed_lag_smoother = true;
    const double scan_period = params.scan_time.scan_duration.value_or(0.1);
    double trajectory_span = scan_period * static_cast<double>(std::max<size_t>(1, scans.size()));
    if (scans.size() >= 2 && scans.back().stamp > scans.front().stamp) {
      trajectory_span = std::max(trajectory_span, scans.back().stamp - scans.front().stamp + scan_period);
    }
    double lag = std::max(params.smoother_lag, trajectory_span + 1.0);
    const double cap = params.slam_mapping.odometry_smoother_lag_cap;
    if (cap > 0.0) {
      lag = std::min(lag, cap);
    }
    odom_params.smoother_lag = lag;
  } else {
    odom_params.enable_fixed_lag_smoother = params.enable_fixed_lag_smoother;
  }

  glim::OdometryEstimationCT odom(odom_params);
  std::unique_ptr<glim::SubMapping> sub_mapping;
  if (params.slam_mapping.enable_sub_mapping) {
    sub_mapping = std::make_unique<glim::SubMapping>(sub_mapping_params_from_fusion(params));
  }
  std::unique_ptr<glim::GlobalMapping> global_mapping;
  if (params.slam_mapping.enable_global_mapping) {
    global_mapping = std::make_unique<glim::GlobalMapping>(global_mapping_params_from_fusion(params));
  }

  const int odom_target = slam_odometry_target_points(params);
  std::mt19937 subsample_rng(0);

  std::vector<glim::PreprocessedFrame::ConstPtr> export_scans;
  export_scans.reserve(scans.size());
  std::vector<glim::EstimationFrame::ConstPtr> odom_frames_by_scan;
  odom_frames_by_scan.reserve(scans.size());

  for (size_t si = 0; si < scans.size(); ++si) {
    if (on_progress) {
      FusionProgress progress;
      progress.scan_index = static_cast<int>(si + 1);
      progress.total_scans = static_cast<int>(scans.size());
      progress.stage = "slam_odometry";
      progress.scan_stamp = scans[si].stamp;
      on_progress(progress);
    }

    std::string err;
    const auto full_preprocessed = to_preprocessed(scans[si], params, err, false);
    if (!full_preprocessed) {
      result.error = FusionError::InvalidArgument;
      result.message = err;
      return result;
    }

    const auto odom_preprocessed =
      subsample_preprocessed_with_neighbors(*full_preprocessed, odom_target, params.k_neighbors, params.num_threads, subsample_rng);
    if (!odom_preprocessed) {
      result.error = FusionError::InvalidArgument;
      result.message = "too few points after odometry subsampling";
      return result;
    }

    if (params.slam_mapping.export_full_resolution) {
      export_scans.push_back(full_preprocessed);
    } else {
      const auto export_sub = subsample_preprocessed_with_neighbors(
        *full_preprocessed, params.target_num_points, params.k_neighbors, params.num_threads, subsample_rng, false);
      if (!export_sub) {
        result.error = FusionError::InvalidArgument;
        result.message = "too few points after export subsampling";
        return result;
      }
      export_scans.push_back(export_sub);
    }

    std::vector<glim::EstimationFrame::ConstPtr> marginalized;
    const auto odom_frame = odom.insert_frame(odom_preprocessed, marginalized);
    if (!odom_frame) {
      result.error = FusionError::FusionFailed;
      result.message = "odometry insert_frame failed";
      return result;
    }
    odom_frames_by_scan.push_back(odom_frame);

    if (sub_mapping) {
      sub_mapping->insert_frame(odom_frame);
    }
  }

  std::vector<glim::SubMap::Ptr> submaps;
  if (sub_mapping) {
    if (on_progress) {
      FusionProgress progress;
      progress.scan_index = static_cast<int>(scans.size());
      progress.total_scans = static_cast<int>(scans.size());
      progress.stage = "slam_sub_mapping";
      on_progress(progress);
    }
    auto queued = sub_mapping->get_submaps();
    auto tail = sub_mapping->submit_end_of_sequence();
    submaps.insert(submaps.end(), queued.begin(), queued.end());
    submaps.insert(submaps.end(), tail.begin(), tail.end());
  }

  if (submaps.empty()) {
    result.error = FusionError::FusionFailed;
    result.message = "sub-mapping produced no submaps";
    return result;
  }

  std::vector<Eigen::Isometry3d> submap_origin_before_global;
  if (global_mapping) {
    if (on_progress) {
      FusionProgress progress;
      progress.scan_index = static_cast<int>(scans.size());
      progress.total_scans = static_cast<int>(scans.size());
      progress.stage = "slam_global_mapping";
      on_progress(progress);
    }
    for (const auto& submap : submaps) {
      submap_origin_before_global.push_back(submap->T_world_origin);
      global_mapping->insert_submap(submap);
    }
    if (submaps.size() > 1) {
      global_mapping->find_overlapping_submaps(params.slam_mapping.global_min_implicit_loop_overlap);
    }
    global_mapping->optimize();
  }

  struct ScanPose {
    Eigen::Isometry3d begin = Eigen::Isometry3d::Identity();
    Eigen::Isometry3d end = Eigen::Isometry3d::Identity();
  };
  std::vector<ScanPose> scan_poses(scans.size());
  std::vector<bool> pose_from_submap(scans.size(), false);

  for (size_t submap_idx = 0; submap_idx < submaps.size(); ++submap_idx) {
    const auto& submap = submaps[submap_idx];
    Eigen::Isometry3d global_delta = Eigen::Isometry3d::Identity();
    if (global_mapping && submap_idx < submap_origin_before_global.size()) {
      global_delta = submap->T_world_origin * submap_origin_before_global[submap_idx].inverse();
    }

    for (const auto& frame : submap->frames) {
      if (!frame || frame->id < 0 || static_cast<size_t>(frame->id) >= scan_poses.size()) {
        continue;
      }
      scan_poses[frame->id].begin = global_delta * frame->T_world_sensor();
      scan_poses[frame->id].end = global_delta * frame->T_world_lidar_end;
      pose_from_submap[frame->id] = true;
    }
  }

  for (size_t si = 0; si < scans.size(); ++si) {
    if (pose_from_submap[si] || si >= odom_frames_by_scan.size() || !odom_frames_by_scan[si]) {
      continue;
    }
    const auto& odom_frame = odom_frames_by_scan[si];
    scan_poses[si].begin = odom_frame->T_world_sensor();
    scan_poses[si].end = odom_frame->T_world_lidar_end;
  }

  OutputVoxelMap output_voxels;
  const int nt = fusion_effective_num_threads(params.num_threads);
#ifdef _OPENMP
  std::vector<OutputVoxelMap> scan_voxel_maps(scans.size());
#pragma omp parallel for schedule(dynamic) num_threads(nt)
  for (int si = 0; si < static_cast<int>(scans.size()); ++si) {
    const auto& raw = export_scans[si];
    if (!raw || raw->full_scan_duration <= 0.0) {
      continue;
    }
    accumulate_deskewed_scan(
      *raw,
      scan_poses[si].begin,
      scan_poses[si].end,
      raw->full_scan_duration,
      params.voxel_resolution,
      params.min_distance,
      scan_voxel_maps[si],
      1);
  }
  for (auto& local : scan_voxel_maps) {
    for (auto& [key, value] : local) {
      output_voxels[key] = value;
    }
  }
#else
  for (size_t si = 0; si < scans.size(); ++si) {
    const auto& raw = export_scans[si];
    if (!raw || raw->full_scan_duration <= 0.0) {
      continue;
    }
    accumulate_deskewed_scan(
      *raw,
      scan_poses[si].begin,
      scan_poses[si].end,
      raw->full_scan_duration,
      params.voxel_resolution,
      params.min_distance,
      output_voxels,
      params.num_threads);
  }
#endif

  for (size_t si = 0; si < scans.size(); ++si) {
    result.scan_poses.push_back(scan_poses[si].begin);
  }

  result.cloud = dense_from_voxels(output_voxels);
  result.error = FusionError::Ok;
  return result;
}

}  // namespace glim_cloud_fusion::detail
