#include "integrator_sampling.hpp"

#include <algorithm>
#include <cmath>
#include <iterator>
#include <unordered_map>

namespace glim_cloud_fusion::detail {

namespace {

struct GridKey {
  int x, y, z;
  bool operator==(const GridKey& o) const { return x == o.x && y == o.y && z == o.z; }
};

struct GridKeyHash {
  size_t operator()(const GridKey& k) const {
    return (static_cast<size_t>(k.x) * 73856093u) ^ (static_cast<size_t>(k.y) * 19349663u) ^ (static_cast<size_t>(k.z) * 83492791u);
  }
};

}  // namespace

std::vector<int> integrator_randomgrid_subsample_indices(
  const std::vector<Eigen::Vector4d>& points,
  const std::vector<double>& times,
  int target_num_points,
  std::mt19937& mt) {
  const int n = static_cast<int>(points.size());
  if (n == 0) {
    return {};
  }
  if (target_num_points <= 0 || n <= target_num_points) {
    std::vector<int> all(n);
    for (int i = 0; i < n; ++i) {
      all[i] = i;
    }
    return all;
  }

  const double sampling_rate = static_cast<double>(target_num_points) / static_cast<double>(n);
  if (sampling_rate >= 0.99) {
    std::vector<int> all(n);
    for (int i = 0; i < n; ++i) {
      all[i] = i;
    }
    return all;
  }

  std::vector<int> time_order(n);
  for (int i = 0; i < n; ++i) {
    time_order[i] = i;
  }
  std::sort(time_order.begin(), time_order.end(), [&](int a, int b) { return times[a] < times[b]; });

  std::unordered_map<GridKey, std::vector<int>, GridKeyHash> voxelmap;
  voxelmap.reserve(static_cast<size_t>(n * sampling_rate));

  for (int i = 0; i < n; ++i) {
    const int idx = time_order[i];
    const auto& p = points[idx];
    const GridKey key{
      static_cast<int>(std::floor(p.x() / kIntegratorRandomGridVoxel)),
      static_cast<int>(std::floor(p.y() / kIntegratorRandomGridVoxel)),
      static_cast<int>(std::floor(p.z() / kIntegratorRandomGridVoxel))};
    voxelmap[key].push_back(idx);
  }

  const int points_per_voxel =
    static_cast<int>(std::ceil((sampling_rate * static_cast<double>(n)) / static_cast<double>(voxelmap.size())));
  const int max_num_points = static_cast<int>(std::ceil(static_cast<double>(n) * sampling_rate * 1.2));

  std::vector<int> indices;
  indices.reserve(static_cast<size_t>(max_num_points));

  for (const auto& [_, voxel_indices] : voxelmap) {
    if (static_cast<int>(voxel_indices.size()) <= points_per_voxel) {
      indices.insert(indices.end(), voxel_indices.begin(), voxel_indices.end());
    } else {
      std::sample(
        voxel_indices.begin(),
        voxel_indices.end(),
        std::back_inserter(indices),
        static_cast<std::ptrdiff_t>(points_per_voxel),
        mt);
    }
  }

  if (static_cast<int>(indices.size()) > max_num_points) {
    std::vector<int> sub(static_cast<size_t>(max_num_points));
    std::sample(indices.begin(), indices.end(), sub.begin(), static_cast<std::ptrdiff_t>(max_num_points), mt);
    indices = std::move(sub);
  }

  std::sort(indices.begin(), indices.end());
  return indices;
}

}  // namespace glim_cloud_fusion::detail
