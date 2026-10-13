#include "fusion_parallel.hpp"

#include <algorithm>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace glim_cloud_fusion::detail {

int fusion_effective_num_threads(int requested) {
#ifdef _OPENMP
  if (requested <= 0) {
    return std::max(1, omp_get_max_threads());
  }
  return requested;
#else
  return 1;
#endif
}

void parallel_knn_search(
  gtsam_points::KdTree& tree,
  const Eigen::Vector4d* points,
  int n,
  int k,
  std::vector<int>& neighbors,
  int num_threads) {
  neighbors.resize(static_cast<size_t>(n) * k);
#ifdef _OPENMP
#pragma omp parallel for num_threads(num_threads) schedule(guided, 8)
#endif
  for (int i = 0; i < n; ++i) {
    std::vector<size_t> k_idx(static_cast<size_t>(k));
    std::vector<double> k_sq(static_cast<size_t>(k));
    tree.knn_search(points[i].data(), k, k_idx.data(), k_sq.data());
    std::copy(k_idx.begin(), k_idx.end(), neighbors.begin() + static_cast<size_t>(i) * k);
  }
}

}  // namespace glim_cloud_fusion::detail
