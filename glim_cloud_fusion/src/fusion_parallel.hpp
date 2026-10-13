#pragma once

#include <vector>

#include <Eigen/Core>
#include <gtsam_points/ann/kdtree.hpp>

namespace glim_cloud_fusion::detail {

int fusion_effective_num_threads(int requested);

void parallel_knn_search(
  gtsam_points::KdTree& tree,
  const Eigen::Vector4d* points,
  int n,
  int k,
  std::vector<int>& neighbors,
  int num_threads);

}  // namespace glim_cloud_fusion::detail
