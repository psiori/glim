#pragma once

#include <functional>
#include <optional>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>

namespace glim_cloud_fusion {

enum class FusionError { Ok = 0, InvalidArgument, EmptyInput, FusionFailed };

enum class FusionBackend { CtIntegrator = 0, GlimSlam = 1 };

enum class TimeUnit { Seconds, Nanoseconds };
enum class TimeOrigin { FirstPoint, ScanStart };

struct ScanTimeParams {
  TimeUnit time_unit = TimeUnit::Seconds;
  TimeOrigin time_origin = TimeOrigin::FirstPoint;
  std::optional<double> scan_duration;
};

struct LidarScan {
  double stamp = 0.0;
  std::vector<Eigen::Vector4d, Eigen::aligned_allocator<Eigen::Vector4d>> points;
  std::vector<double> times;
  std::vector<double> intensities;
  std::vector<bool> keep;  ///< If empty, keep all finite points
  int width = 0;
  int height = 0;
};

struct FusionParams {
  FusionBackend backend = FusionBackend::CtIntegrator;

  /// Post-fusion output downsampling (`voxel_merge`); independent of map resolutions.
  double voxel_resolution = 0.05;
  int num_threads = 4;
  double max_correspondence_distance = 1.0;
  /// Neighbors per point for covariance estimation in CT-GICP preprocessing.
  int k_neighbors = 40;
  ScanTimeParams scan_time;

  /// iVox scan-to-model map (inside `OdometryEstimationCT`).
  double ivox_resolution = 1.0;
  double ivox_min_points_dist = 0.1;
  int ivox_lru_thresh = 200;

  /// CT-GICP graph weights and LM iterations.
  double location_consistency_inf_scale = 1e-3;
  double constant_velocity_inf_scale = 1e3;
  int lm_max_iterations = 8;

  /// Fixed-lag pose smoother (iSAM2).
  double smoother_lag = 1.0;
  bool use_isam2_dogleg = false;
  double isam2_relinearize_skip = 1.0;
  double isam2_relinearize_thresh = 0.1;

  /// Marginal covariances on latest frame (debug; does not change fused cloud).
  bool compute_covs = false;

  /// iSAM2 fixed-lag smoother (off by default for batch fusion; snapshots use per-scan LM poses).
  bool enable_fixed_lag_smoother = false;

  /// CT integrator (`CtIntegrator` backend): per-scan point cap after voxelized subsampling.
  int target_num_points = 10000;
  /// Drop accumulated points closer than this to the sensor origin (m).
  double min_distance = 1.0;
  /// Scan-to-model iVox leaf size during integrator fusion (m).
  double integrator_ivox_resolution = 0.05;
  int integrator_ivox_lru_thresh = 100;
  /// Outer LM repeats per scan (vlcal preprocess uses 3).
  int integrator_lm_outer_loops = 3;
};

struct DensePointCloud {
  std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> points;
  std::vector<double> intensities;
};

struct FusionProgress {
  int scan_index = 0;
  int total_scans = 0;
  std::string stage;
  /// Partial merged cloud for live preview (may be empty if not yet available).
  DensePointCloud preview_cloud;
};

struct FusionResult {
  FusionError error = FusionError::Ok;
  std::string message;
  DensePointCloud cloud;
  std::vector<Eigen::Isometry3d, Eigen::aligned_allocator<Eigen::Isometry3d>> scan_poses;
};

using FusionCallback = std::function<void(const FusionProgress&)>;

FusionResult fuse_scans(const std::vector<LidarScan>& scans, const FusionParams& params, FusionCallback on_progress = nullptr);

/// Normalize per-point times to seconds from the chosen origin. Returns false on invalid input.
bool normalize_lidar_scan_times(
  const LidarScan& scan,
  const ScanTimeParams& params,
  std::vector<double>& out_times,
  double& out_duration,
  std::string& error_message);

inline double scan_time_alpha(double t, double scan_duration) {
  return t / std::max(1e-9, scan_duration);
}

}  // namespace glim_cloud_fusion
