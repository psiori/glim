#include "fusion_detail.hpp"

#include <random>
#include <unordered_map>

#include <gtsam/geometry/Pose3.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/slam/PriorFactor.h>

#include <glim/common/cloud_covariance_estimation.hpp>
#include <gtsam_points/ann/ivox.hpp>
#include <gtsam_points/ann/kdtree.hpp>
#include <gtsam_points/factors/integrated_ct_gicp_factor.hpp>
#include <gtsam_points/types/point_cloud_cpu.hpp>

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

std::vector<int> subsample_indices(int n, int target, std::mt19937& mt) {
  std::vector<int> indices(n);
  for (int i = 0; i < n; ++i) {
    indices[i] = i;
  }
  if (n <= target || target <= 0) {
    return indices;
  }
  std::vector<int> out(target);
  std::sample(indices.begin(), indices.end(), out.begin(), target, mt);
  std::sort(out.begin(), out.end());
  return out;
}

gtsam_points::PointCloudCPU::Ptr build_source_cloud(
  const glim::PreprocessedFrame& raw,
  const std::vector<int>& sample_indices,
  glim::CloudCovarianceEstimation& cov_est,
  int num_threads) {
  const int n = static_cast<int>(sample_indices.size());
  gtsam_points::PointCloudCPU::Ptr cloud(new gtsam_points::PointCloudCPU());
  cloud->points_storage.resize(n);
  cloud->times_storage.resize(n);
  for (int i = 0; i < n; ++i) {
    const int src = sample_indices[i];
    cloud->points_storage[i] = raw.points[src];
    cloud->times_storage[i] = raw.times[src];
  }
  cloud->points = cloud->points_storage.data();
  cloud->times = cloud->times_storage.data();

  const int k = std::max(2, raw.k_neighbors);
  std::vector<int> neighbors(n * k, 0);
  gtsam_points::KdTree tree(cloud->points, n);
  for (int i = 0; i < n; ++i) {
    std::vector<size_t> k_idx(k);
    std::vector<double> k_sq(k);
    tree.knn_search(cloud->points[i].data(), k, k_idx.data(), k_sq.data());
    std::copy(k_idx.begin(), k_idx.end(), neighbors.begin() + i * k);
  }
  cloud->normals_storage.resize(n);
  cloud->covs_storage.resize(n);
  cov_est.estimate(cloud->points_storage, neighbors, cloud->normals_storage, cloud->covs_storage);
  cloud->normals = cloud->normals_storage.data();
  cloud->covs = cloud->covs_storage.data();
  return cloud;
}

void accumulate_scan(
  const glim::PreprocessedFrame& raw,
  const std::vector<int>& time_order,
  const gtsam::Pose3& T_begin,
  const gtsam::Pose3& T_end,
  double scan_duration,
  double voxel_resolution,
  double min_distance,
  OutputVoxelMap& voxels) {
  double last_t = -1.0;
  gtsam::Pose3 T = T_begin;
  const double time_eps = 1e-4;
  const double inv_dur = 1.0 / std::max(1e-9, scan_duration);

  for (const int idx : time_order) {
    const double t = raw.times[idx] * inv_dur;
    if (t - last_t > time_eps) {
      last_t = t;
      T = T_begin.interpolateRt(T_end, t);
    }
    const Eigen::Vector4d pt = T.matrix() * raw.points[idx];
    if (pt.head<3>().norm() < min_distance) {
      continue;
    }
    const OutputVoxelKey key{
      static_cast<int>(std::floor(pt.x() / voxel_resolution)),
      static_cast<int>(std::floor(pt.y() / voxel_resolution)),
      static_cast<int>(std::floor(pt.z() / voxel_resolution))};
    const double intensity = raw.intensities.empty() ? 0.0 : raw.intensities[idx];
    voxels[key] = Eigen::Vector4d(pt.x(), pt.y(), pt.z(), intensity);
  }
}

DensePointCloud dense_from_output_voxels(const OutputVoxelMap& voxels) {
  DensePointCloud out;
  out.points.reserve(voxels.size());
  out.intensities.reserve(voxels.size());
  for (const auto& [_, v] : voxels) {
    out.points.emplace_back(v.x(), v.y(), v.z());
    out.intensities.push_back(v.w());
  }
  return out;
}

}  // namespace

FusionResult fuse_scans_ct_integrator(
  const std::vector<LidarScan>& scans,
  const FusionParams& params,
  FusionCallback on_progress) {
  FusionResult result;
  glim::CloudCovarianceEstimation cov_est(params.num_threads);

  auto target_ivox = std::make_shared<gtsam_points::iVox>(params.integrator_ivox_resolution);
  target_ivox->voxel_insertion_setting().set_min_dist_in_cell(params.ivox_min_points_dist);
  target_ivox->set_lru_horizon(params.integrator_ivox_lru_thresh);
  target_ivox->set_neighbor_voxel_mode(1);

  gtsam::Pose3 last_begin = gtsam::Pose3();
  gtsam::Pose3 last_end = gtsam::Pose3();
  OutputVoxelMap output_voxels;
  std::mt19937 mt(0);

  const size_t preview_stride = std::max<size_t>(1, scans.size() / 10);
  const int outer_loops = std::max(1, params.integrator_lm_outer_loops);
  const int lm_iters = std::max(1, params.lm_max_iterations);

  for (size_t si = 0; si < scans.size(); ++si) {
    std::string err;
    const auto raw = to_preprocessed(scans[si], params, err);
    if (!raw) {
      result.error = FusionError::InvalidArgument;
      result.message = err;
      return result;
    }

    const std::vector<int> time_order = sort_indices_by_time(raw->times);
    if (raw->full_scan_duration <= 0.0 || !std::isfinite(raw->full_scan_duration)) {
      result.error = FusionError::InvalidArgument;
      result.message = "scan_duration is required";
      return result;
    }
    const double scan_duration = raw->full_scan_duration;

    if (!target_ivox->has_points()) {
      const std::vector<int> sample = subsample_indices(static_cast<int>(raw->points.size()), params.target_num_points, mt);
      auto cloud = build_source_cloud(*raw, sample, cov_est, params.num_threads);
      target_ivox->insert(*cloud);
      last_begin = gtsam::Pose3();
      last_end = gtsam::Pose3();
      accumulate_scan(*raw, time_order, last_begin, last_end, scan_duration, params.voxel_resolution, params.min_distance, output_voxels);
      result.scan_poses.push_back(Eigen::Isometry3d::Identity());
    } else {
      const std::vector<int> sample = subsample_indices(static_cast<int>(raw->points.size()), params.target_num_points, mt);
      auto source = build_source_cloud(*raw, sample, cov_est, params.num_threads);

      const gtsam::Vector6 pred_v = gtsam::Pose3::Logmap(last_begin.between(last_end)) / std::max(1e-9, scan_duration);
      const gtsam::Pose3 pred_begin = last_end;
      const gtsam::Pose3 pred_end = last_end * gtsam::Pose3::Expmap(0.75 * pred_v * scan_duration);

      gtsam::Values values;
      values.insert(0, pred_begin);
      values.insert(1, pred_end);

      gtsam::NonlinearFactorGraph graph;
      auto factor = std::make_shared<gtsam_points::IntegratedCT_GICPFactor_<gtsam_points::iVox, gtsam_points::PointCloud>>(
        0, 1, target_ivox, source, target_ivox);
      factor->set_num_threads(params.num_threads);
      factor->set_max_correspondence_distance(params.max_correspondence_distance);
      factor->set_time_span(0.0, scan_duration);
      graph.add(factor);
      graph.emplace_shared<gtsam::PriorFactor<gtsam::Pose3>>(
        0, pred_begin, gtsam::noiseModel::Isotropic::Precision(6, 1e3));
      graph.emplace_shared<gtsam::BetweenFactor<gtsam::Pose3>>(
        0, 1, gtsam::Pose3(), gtsam::noiseModel::Isotropic::Precision(6, params.constant_velocity_inf_scale));

      gtsam::LevenbergMarquardtParams lm_params;
      lm_params.setMaxIterations(lm_iters);

      for (int outer = 0; outer < outer_loops; ++outer) {
        values = gtsam::LevenbergMarquardtOptimizer(graph, values, lm_params).optimize();
      }

      last_begin = values.at<gtsam::Pose3>(0);
      last_end = values.at<gtsam::Pose3>(1);

      const auto deskewed = factor->deskewed_source_points(values, true);
      const int n = static_cast<int>(deskewed.size());
      const int k = std::max(2, raw->k_neighbors);
      std::vector<int> nn(n * k, 0);
      gtsam_points::KdTree tree(deskewed.data(), n);
      for (int i = 0; i < n; ++i) {
        std::vector<size_t> k_idx(k);
        std::vector<double> k_sq(k);
        tree.knn_search(deskewed[i].data(), k, k_idx.data(), k_sq.data());
        std::copy(k_idx.begin(), k_idx.end(), nn.begin() + i * k);
      }
      source->normals_storage.resize(n);
      source->covs_storage.resize(n);
      cov_est.estimate(deskewed, nn, source->normals_storage, source->covs_storage);
      for (int i = 0; i < n; ++i) {
        source->points_storage[i] = deskewed[i];
      }
      source->points = source->points_storage.data();
      source->covs = source->covs_storage.data();
      target_ivox->insert(*source);

      accumulate_scan(*raw, time_order, last_begin, last_end, scan_duration, params.voxel_resolution, params.min_distance, output_voxels);
      result.scan_poses.emplace_back(last_begin.matrix());
    }

    if (on_progress) {
      FusionProgress progress;
      progress.scan_index = static_cast<int>(si + 1);
      progress.total_scans = static_cast<int>(scans.size());
      progress.stage = "integrator";
      if (si == 0 || si + 1 == scans.size() || (si + 1) % preview_stride == 0) {
        progress.preview_cloud = dense_from_output_voxels(output_voxels);
      }
      on_progress(progress);
    }
  }

  result.cloud = dense_from_output_voxels(output_voxels);
  result.error = FusionError::Ok;
  return result;
}

}  // namespace glim_cloud_fusion::detail
