#include <glim_cloud_fusion/fusion.hpp>

#include <cmath>
#include <unordered_map>

#include <glim/mapping/callbacks.hpp>
#include <glim/mapping/global_mapping.hpp>
#include <glim/mapping/sub_mapping.hpp>
#include <glim/odometry/callbacks.hpp>
#include <glim/odometry/odometry_estimation_ct.hpp>
#include <glim/preprocess/preprocessed_frame.hpp>
#include <glim/util/raw_points.hpp>
#include <gtsam_points/ann/kdtree.hpp>
#include <gtsam_points/types/point_cloud_cpu.hpp>

namespace glim_cloud_fusion {

namespace {

struct VoxelKey {
  int x, y, z;
  bool operator==(const VoxelKey& o) const { return x == o.x && y == o.y && z == o.z; }
};

struct VoxelKeyHash {
  size_t operator()(const VoxelKey& k) const {
    return (static_cast<size_t>(k.x) * 73856093u) ^ (static_cast<size_t>(k.y) * 19349663u) ^ (static_cast<size_t>(k.z) * 83492791u);
  }
};

struct VoxelAccum {
  Eigen::Vector3d point = Eigen::Vector3d::Zero();
  double intensity = 0.0;
  int count = 0;
};

glim::PreprocessedFrame::Ptr to_preprocessed(const LidarScan& scan, const FusionParams& params, std::string& err) {
  std::vector<double> norm_times;
  double scan_duration = 0.0;
  if (!normalize_lidar_scan_times(scan, params.scan_time, norm_times, scan_duration, err)) {
    return nullptr;
  }

  auto frame = glim::PreprocessedFrame::Ptr(new glim::PreprocessedFrame);
  frame->stamp = scan.stamp;
  frame->full_scan_duration = scan_duration;
  frame->scan_end_time = scan.stamp + norm_times.back();

  const bool use_mask = !scan.keep.empty();
  for (size_t i = 0; i < scan.points.size(); ++i) {
    if (!scan.points[i].array().isFinite().all()) {
      continue;
    }
    if (use_mask && (i >= scan.keep.size() || !scan.keep[i])) {
      continue;
    }
    frame->points.push_back(scan.points[i]);
    frame->times.push_back(norm_times[i]);
    if (!scan.intensities.empty()) {
      frame->intensities.push_back(scan.intensities[i]);
    }
  }

  if (frame->points.size() < 10) {
    err = "too few points after masking";
    return nullptr;
  }

  frame->k_neighbors = 20;
  frame->neighbors.resize(frame->points.size() * frame->k_neighbors, 0);
  if (frame->points.size() >= frame->k_neighbors) {
    gtsam_points::KdTree tree(frame->points.data(), frame->points.size());
    for (int i = 0; i < static_cast<int>(frame->points.size()); ++i) {
      std::vector<size_t> k_indices(frame->k_neighbors);
      std::vector<double> k_sq_dists(frame->k_neighbors);
      tree.knn_search(frame->points[i].data(), frame->k_neighbors, k_indices.data(), k_sq_dists.data());
      std::copy(k_indices.begin(), k_indices.end(), frame->neighbors.begin() + i * frame->k_neighbors);
    }
  }
  return frame;
}

DensePointCloud voxel_merge(
  const std::vector<Eigen::Vector4d, Eigen::aligned_allocator<Eigen::Vector4d>>& points,
  const std::vector<double>& intensities,
  double voxel_resolution) {
  std::unordered_map<VoxelKey, VoxelAccum, VoxelKeyHash> voxels;
  for (size_t i = 0; i < points.size(); ++i) {
    const Eigen::Vector3d p = points[i].head<3>();
    const VoxelKey key{
      static_cast<int>(std::floor(p.x() / voxel_resolution)),
      static_cast<int>(std::floor(p.y() / voxel_resolution)),
      static_cast<int>(std::floor(p.z() / voxel_resolution))};
    auto& v = voxels[key];
    v.point += p;
    if (!intensities.empty()) {
      v.intensity += intensities[i];
    }
    v.count++;
  }

  DensePointCloud out;
  for (const auto& [_, v] : voxels) {
    out.points.push_back(v.point / static_cast<double>(v.count));
    out.intensities.push_back(v.count > 0 ? v.intensity / v.count : 0.0);
  }
  return out;
}

}  // namespace

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

  if (params.time_origin == TimeOrigin::FirstPoint) {
    const double t0 = *std::min_element(out_times.begin(), out_times.end());
    for (auto& t : out_times) {
      t -= t0;
    }
    out_duration = out_times.back();
    if (params.scan_duration.has_value()) {
      out_duration = *params.scan_duration;
    }
  } else {
    if (!params.scan_duration.has_value() || *params.scan_duration <= 0.0) {
      err = "scan_duration required for TimeOrigin::ScanStart";
      return false;
    }
    out_duration = *params.scan_duration;
  }
  return out_duration > 0.0;
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
    const auto frame = to_preprocessed(scans.front(), params, err);
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
    result.cloud = voxel_merge(pts, intensities, params.voxel_resolution);
    result.scan_poses.push_back(Eigen::Isometry3d::Identity());
    return result;
  }

  glim::OdometryEstimationCTParams odom_params;
  odom_params.num_threads = params.num_threads;
  odom_params.max_correspondence_distance = params.max_correspondence_distance;
  glim::OdometryEstimationCT odom(odom_params);

  glim::SubMappingParams sub_params;
  sub_params.enable_imu = false;
  sub_params.enable_gpu = false;
  sub_params.registration_error_factor_type = "VGICP";
  glim::SubMapping sub_mapping(sub_params);

  glim::GlobalMappingParams global_params;
  global_params.enable_imu = false;
  global_params.enable_gpu = false;
  glim::GlobalMapping global_mapping(global_params);

  struct Snapshot {
    std::vector<Eigen::Vector4d, Eigen::aligned_allocator<Eigen::Vector4d>> points;
    std::vector<double> intensities;
    Eigen::Isometry3d T_world_begin = Eigen::Isometry3d::Identity();
  };
  std::vector<Snapshot> snapshots;

  for (size_t si = 0; si < scans.size(); ++si) {
    if (on_progress) {
      on_progress(FusionProgress{static_cast<int>(si), static_cast<int>(scans.size()), "odometry"});
    }
    std::string err;
    const auto preprocessed = to_preprocessed(scans[si], params, err);
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

    for (const auto& marg : marginalized) {
      sub_mapping.insert_frame(marg);
    }
    for (const auto& submap : sub_mapping.get_submaps()) {
      global_mapping.insert_submap(submap);
    }
  }

  for (const auto& submap : sub_mapping.submit_end_of_sequence()) {
    global_mapping.insert_submap(submap);
  }

  if (on_progress) {
    on_progress(FusionProgress{static_cast<int>(scans.size()), static_cast<int>(scans.size()), "optimize"});
  }
  global_mapping.optimize();

  std::vector<Eigen::Vector4d, Eigen::aligned_allocator<Eigen::Vector4d>> merged_pts;
  std::vector<double> merged_intensities;
  for (const auto& snap : snapshots) {
    const Eigen::Isometry3d T = snap.T_world_begin;
    for (size_t i = 0; i < snap.points.size(); ++i) {
      merged_pts.push_back(T * snap.points[i]);
      merged_intensities.push_back(snap.intensities[i]);
    }
  }

  result.cloud = voxel_merge(merged_pts, merged_intensities, params.voxel_resolution);
  result.error = FusionError::Ok;
  return result;
}

}  // namespace glim_cloud_fusion
