#pragma once

#include <functional>
#include <optional>
#include <ostream>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>

namespace glim_cloud_fusion {

enum class FusionError { Ok = 0, InvalidArgument, EmptyInput, FusionFailed };

enum class FusionBackend { CtIntegrator = 0, GlimSlam = 1 };

enum class TimeUnit { Seconds, Nanoseconds };
/// How per-point times in `LidarScan::times` are interpreted before deskew.
enum class TimeOrigin {
  /// Subtract the minimum per-point time in the stored scan (first point in the PCD).
  FirstPoint,
  /// Times are already relative to a fixed sweep anchor (e.g. rotation start). Do not subtract the first point;
  /// required when the saved scan omits angular sections but timestamps still count from rotation start.
  ScanStart
};

struct ScanTimeParams {
  TimeUnit time_unit = TimeUnit::Nanoseconds;
  TimeOrigin time_origin = TimeOrigin::ScanStart;
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

/// GLIM sub-mapping + global mapping options for `FusionBackend::GlimSlam`.
struct SlamMappingParams {
  bool enable_sub_mapping = true;
  bool enable_global_mapping = true;
  /// Run CT odometry fixed-lag ISAM2 before sub-mapping (poses fed to the mapper).
  bool enable_odometry_smoother = true;
  /// Finalize and optimize a submap once this many **keyframes** are accumulated (not scans).
  /// When the cap is hit during the scan loop, that submap is optimized immediately; global mapping still runs once at the end.
  int submap_max_num_keyframes = 500;
  /// `OVERLAP` or `DISPLACEMENT`: when to add another keyframe **inside** the active submap (does not split submaps).
  std::string keyframe_update_strategy = "OVERLAP";
  /// OVERLAP strategy: add a keyframe when scan/keyframe overlap drops below this (0–1). Higher → fewer keyframes.
  double max_keyframe_overlap = 0.6;
  double keyframe_update_interval_trans = 0.25;
  double keyframe_update_interval_rot = 0.15;
  double submap_downsample_resolution = 0.05;
  double global_min_implicit_loop_overlap = 0.2;
  double global_max_implicit_loop_distance = 50.0;
  /// Cap CT odometry fixed-lag ISAM2 window (s). 0 disables the cap (full trajectory lag when smoother is on).
  double odometry_smoother_lag_cap = 8.0;
  /// CT odometry registration point cap per scan (0 = use `FusionParams::target_num_points`).
  int odometry_target_num_points = 0;
  /// If false, deskew merge subsamples each scan to `target_num_points` (faster export).
  bool export_full_resolution = true;
};

struct FusionParams {
  FusionBackend backend = FusionBackend::CtIntegrator;

  /// Integrator output voxel hash / SLAM `voxel_merge` cell size (m). vlcal `DynamicPointCloudIntegratorParams::voxel_resolution`.
  double voxel_resolution = 0.05;
  int num_threads = 16;
  /// CT-GICP max correspondence (m). vlcal factor default when unset.
  double max_correspondence_distance = 1.0;
  /// Neighbors per point for covariance estimation in CT-GICP preprocessing.
  int k_neighbors = 20;
  ScanTimeParams scan_time;

  /// iVox scan-to-model map (inside `OdometryEstimationCT`).
  double ivox_resolution = 1.0;
  double ivox_min_points_dist = 0.1;
  int ivox_lru_thresh = 200;

  /// CT-GICP graph weights and LM iterations.
  double location_consistency_inf_scale = 1e-3;
  double constant_velocity_inf_scale = 1e5;
  int lm_max_iterations = 10;

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
  /// Drop points closer than this to the sensor origin in scanner frame before deskew/world transform (m). Not iVox insertion spacing.
  double min_distance = 1.0;
  /// Scan-to-model iVox voxel leaf size (m). vlcal `iVox(1.0, …)`.
  double integrator_ivox_resolution = 1.0;
  /// Minimum distance between points inserted into the same integrator iVox cell (m). vlcal `iVox(…, 0.05, …)`.
  double integrator_ivox_min_points_dist = 0.05;
  int integrator_ivox_lru_thresh = 100;
  /// Outer LM repeats per scan (vlcal preprocess uses 3).
  int integrator_lm_outer_loops = 3;

  /// Per-scan CT integrator stats on progress callbacks (deskew span, LM cost, poses).
  bool integrator_debug = false;

  SlamMappingParams slam_mapping;
};

struct FusionIntegratorDiagnostics {
  bool is_seed_scan = false;
  int num_raw_points = 0;
  int num_subsample_points = 0;
  double time_min_s = 0.0;
  double time_max_s = 0.0;
  double alpha_min = 0.0;
  double alpha_max = 0.0;
  std::size_t target_map_points = 0;
  std::size_t output_voxel_count = 0;
  double lm_error_initial = 0.0;
  double lm_error_final = 0.0;
  /// Motion within the scan (optimized begin vs end).
  double scan_begin_end_trans_m = 0.0;
  double scan_begin_end_rot_deg = 0.0;
  /// End pose in the integrator odometry frame.
  double odom_end_trans_m = 0.0;
  double odom_end_rot_deg = 0.0;
};

struct DensePointCloud {
  std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> points;
  std::vector<double> intensities;
};

struct FusionProgress {
  int scan_index = 0;
  int total_scans = 0;
  std::string stage;
  /// `LidarScan::stamp` for the scan currently being fused (seconds).
  double scan_stamp = 0.0;
  /// Partial merged cloud for live preview (may be empty if not yet available).
  DensePointCloud preview_cloud;
  /// Set when `FusionParams::integrator_debug` and stage is `integrator` or `integrator_seed`.
  std::optional<FusionIntegratorDiagnostics> integrator;
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

void print_fusion_integrator_diagnostics(std::ostream& out, int scan_index, const FusionIntegratorDiagnostics& d);

}  // namespace glim_cloud_fusion
