#include "fusion_detail.hpp"

#include "fusion_parallel.hpp"

#include <algorithm>
#include <cmath>
#include <unordered_map>

#include <glim/odometry/odometry_estimation_ct.hpp>
#include <gtsam_points/ann/kdtree.hpp>

namespace glim_cloud_fusion::detail {

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

}  // namespace

glim::PreprocessedFrame::Ptr to_preprocessed(
  const LidarScan& scan,
  const FusionParams& params,
  std::string& err,
  const bool compute_full_scan_neighbors) {
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

  const int k_neighbors = std::max(2, params.k_neighbors);
  frame->k_neighbors = k_neighbors;
  if (compute_full_scan_neighbors && static_cast<int>(frame->points.size()) >= frame->k_neighbors) {
    const int n = static_cast<int>(frame->points.size());
    const int num_threads = fusion_effective_num_threads(params.num_threads);
    gtsam_points::KdTree tree(frame->points.data(), n);
    parallel_knn_search(tree, frame->points.data(), n, k_neighbors, frame->neighbors, num_threads);
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

glim::OdometryEstimationCTParams odometry_params_from_fusion(const FusionParams& params) {
  glim::OdometryEstimationCTParams odom(glim::OdometryEstimationCTParams::FromCaller{});
  odom.num_threads = params.num_threads;
  odom.max_correspondence_distance = params.max_correspondence_distance;
  odom.ivox_resolution = params.ivox_resolution;
  odom.ivox_min_points_dist = params.ivox_min_points_dist;
  odom.ivox_lru_thresh = params.ivox_lru_thresh;
  odom.location_consistency_inf_scale = params.location_consistency_inf_scale;
  odom.constant_velocity_inf_scale = params.constant_velocity_inf_scale;
  odom.lm_max_iterations = params.lm_max_iterations;
  odom.smoother_lag = params.smoother_lag;
  odom.use_isam2_dogleg = params.use_isam2_dogleg;
  odom.isam2_relinearize_skip = params.isam2_relinearize_skip;
  odom.isam2_relinearize_thresh = params.isam2_relinearize_thresh;
  odom.compute_covs = params.compute_covs;
  odom.enable_fixed_lag_smoother = params.enable_fixed_lag_smoother;
  return odom;
}

}  // namespace glim_cloud_fusion::detail
