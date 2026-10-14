#pragma once

#include <random>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <glim/preprocess/preprocessed_frame.hpp>
#include <glim/mapping/global_mapping.hpp>
#include <glim/mapping/sub_mapping.hpp>
#include <glim/odometry/odometry_estimation_ct.hpp>
#include <glim_cloud_fusion/fusion.hpp>

namespace glim_cloud_fusion::detail {

/// When `compute_full_scan_neighbors` is false (CT integrator), skip kNN on every point;
/// the integrator subsamples and computes neighbors only on the reduced cloud.
glim::PreprocessedFrame::Ptr to_preprocessed(
  const LidarScan& scan,
  const FusionParams& params,
  std::string& err,
  bool compute_full_scan_neighbors = true);

/// Random-grid subsample + kNN (for CT odometry / registration). `full` must not rely on precomputed neighbors.
glim::PreprocessedFrame::Ptr subsample_preprocessed_with_neighbors(
  const glim::PreprocessedFrame& full,
  int target_num_points,
  int k_neighbors,
  int num_threads,
  std::mt19937& mt,
  bool compute_neighbors = true);

DensePointCloud voxel_merge(
  const std::vector<Eigen::Vector4d, Eigen::aligned_allocator<Eigen::Vector4d>>& points,
  const std::vector<double>& intensities,
  double voxel_resolution);

glim::OdometryEstimationCTParams odometry_params_from_fusion(const FusionParams& params);

glim::SubMappingParams sub_mapping_params_from_fusion(const FusionParams& params);
glim::GlobalMappingParams global_mapping_params_from_fusion(const FusionParams& params);

FusionResult fuse_scans_slam_mapping(
  const std::vector<LidarScan>& scans,
  const FusionParams& params,
  FusionCallback on_progress,
  FusionCancellationCheck is_cancelled);

FusionResult fuse_scans_ct_integrator(
  const std::vector<LidarScan>& scans,
  const FusionParams& params,
  FusionCallback on_progress,
  FusionCancellationCheck is_cancelled);

}  // namespace glim_cloud_fusion::detail
