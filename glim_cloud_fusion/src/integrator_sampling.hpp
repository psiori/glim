#pragma once

#include <random>
#include <vector>

#include <Eigen/Core>

namespace glim_cloud_fusion::detail {

/// vlcal `randomgrid_sampling` grid voxel (fixed in `DynamicPointCloudIntegrator`).
inline constexpr double kIntegratorRandomGridVoxel = 0.5;

/// Sort by per-point time, then spatially stratified subsample (vlcal `randomgrid_sampling`).
std::vector<int> integrator_randomgrid_subsample_indices(
  const std::vector<Eigen::Vector4d>& points,
  const std::vector<double>& times,
  int target_num_points,
  std::mt19937& mt);

}  // namespace glim_cloud_fusion::detail
