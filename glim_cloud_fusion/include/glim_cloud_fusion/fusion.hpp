#pragma once

#include <functional>
#include <optional>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>

namespace glim_cloud_fusion {

enum class FusionError { Ok = 0, InvalidArgument, EmptyInput, FusionFailed };

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
  int num_threads = 4;
  double voxel_resolution = 0.05;
  double max_correspondence_distance = 1.0;
  ScanTimeParams scan_time;
};

struct FusionProgress {
  int scan_index = 0;
  int total_scans = 0;
  std::string stage;
};

struct DensePointCloud {
  std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> points;
  std::vector<double> intensities;
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
