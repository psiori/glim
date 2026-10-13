#pragma once

#include <string>
#include <vector>

#include <Eigen/Core>
#include <glim/preprocess/preprocessed_frame.hpp>
#include <glim/odometry/odometry_estimation_ct.hpp>
#include <glim_cloud_fusion/fusion.hpp>

namespace glim_cloud_fusion::detail {

glim::PreprocessedFrame::Ptr to_preprocessed(const LidarScan& scan, const FusionParams& params, std::string& err);

DensePointCloud voxel_merge(
  const std::vector<Eigen::Vector4d, Eigen::aligned_allocator<Eigen::Vector4d>>& points,
  const std::vector<double>& intensities,
  double voxel_resolution);

glim::OdometryEstimationCTParams odometry_params_from_fusion(const FusionParams& params);

FusionResult fuse_scans_ct_integrator(
  const std::vector<LidarScan>& scans,
  const FusionParams& params,
  FusionCallback on_progress);

}  // namespace glim_cloud_fusion::detail
