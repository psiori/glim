#include <glim/odometry/odometry_estimation_ct.hpp>

#include <future>

#include <spdlog/spdlog.h>

#include <gtsam/inference/Symbol.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/slam/BetweenFactor.h>

#include <gtsam_points/ann/ivox.hpp>
#include <gtsam_points/ann/ivox_covariance_estimation.hpp>
#include <gtsam_points/ann/kdtree.hpp>
#include <gtsam_points/types/point_cloud_cpu.hpp>
#include <gtsam_points/factors/integrated_ct_gicp_factor.hpp>
#include <gtsam_points/optimizers/levenberg_marquardt_ext.hpp>
#include <gtsam_points/optimizers/incremental_fixed_lag_smoother_with_fallback.hpp>

#include <glim/util/config.hpp>
#include <glim/common/cloud_covariance_estimation.hpp>
#include <glim/odometry/callbacks.hpp>

#ifdef GTSAM_USE_TBB
#include <tbb/task_arena.h>
#endif

namespace glim {

using gtsam::symbol_shorthand::X;
using gtsam::symbol_shorthand::Y;
using Callbacks = OdometryEstimationCallbacks;

namespace {

void set_odometry_ct_programmatic_defaults(glim::OdometryEstimationCTParams& params) {
  params.num_threads = 4;
  params.max_correspondence_distance = 1.0;
  params.ivox_resolution = 1.0;
  params.ivox_min_points_dist = 0.1;
  params.ivox_lru_thresh = 200;
  params.location_consistency_inf_scale = 1e-3;
  params.constant_velocity_inf_scale = 1e3;
  params.lm_max_iterations = 8;
  params.smoother_lag = 1.0;
  params.use_isam2_dogleg = false;
  params.isam2_relinearize_skip = 1;
  params.isam2_relinearize_thresh = 0.1;
  params.compute_covs = false;
  params.enable_fixed_lag_smoother = true;
}

void overlay_odometry_ct_config(glim::OdometryEstimationCTParams& params, const Config& config) {
  const char* module = "odometry_estimation";
  if (auto v = config.param<int>(module, "num_threads")) {
    params.num_threads = *v;
  }
  if (auto v = config.param<double>(module, "max_correspondence_distance")) {
    params.max_correspondence_distance = *v;
  }
  if (auto v = config.param<double>(module, "ivox_resolution")) {
    params.ivox_resolution = *v;
  }
  if (auto v = config.param<double>(module, "ivox_min_points_dist")) {
    params.ivox_min_points_dist = *v;
  }
  if (auto v = config.param<int>(module, "ivox_lru_thresh")) {
    params.ivox_lru_thresh = *v;
  }
  if (auto v = config.param<double>(module, "location_consistency_inf_scale")) {
    params.location_consistency_inf_scale = *v;
  }
  if (auto v = config.param<double>(module, "constant_velocity_inf_scale")) {
    params.constant_velocity_inf_scale = *v;
  }
  if (auto v = config.param<int>(module, "lm_max_iterations")) {
    params.lm_max_iterations = *v;
  }
  if (auto v = config.param<double>(module, "smoother_lag")) {
    params.smoother_lag = *v;
  }
  if (auto v = config.param<bool>(module, "use_isam2_dogleg")) {
    params.use_isam2_dogleg = *v;
  }
  if (auto v = config.param<int>(module, "isam2_relinearize_skip")) {
    params.isam2_relinearize_skip = *v;
  }
  if (auto v = config.param<double>(module, "isam2_relinearize_thresh")) {
    params.isam2_relinearize_thresh = *v;
  }
  if (auto v = config.param<bool>(module, "compute_covs")) {
    params.compute_covs = *v;
  }
}

Config load_odometry_ct_config_file() {
  const std::string config_dir = GlobalConfig::instance()->param<std::string>("global", "config_path", ".");
  return Config(config_dir + "/config_odometry_ct.json");
}

}  // namespace

OdometryEstimationCTParams::OdometryEstimationCTParams() {
  set_odometry_ct_programmatic_defaults(*this);
  overlay_odometry_ct_config(*this, load_odometry_ct_config_file());
}

OdometryEstimationCTParams::OdometryEstimationCTParams(FromCaller) {
  set_odometry_ct_programmatic_defaults(*this);
}

OdometryEstimationCTParams::~OdometryEstimationCTParams() {}

OdometryEstimationCT::OdometryEstimationCT(const OdometryEstimationCTParams& params) : params(params) {
  covariance_estimation.reset(new CloudCovarianceEstimation(params.num_threads));

  marginalized_cursor = 0;
  target_ivox.reset(new gtsam_points::iVox(params.ivox_resolution));
  target_ivox->voxel_insertion_setting().set_min_dist_in_cell(params.ivox_min_points_dist);
  target_ivox->set_lru_horizon(params.ivox_lru_thresh);
  target_ivox->set_neighbor_voxel_mode(1);

  gtsam::ISAM2Params isam2_params;
  if (params.use_isam2_dogleg) {
    isam2_params.setOptimizationParams(gtsam::ISAM2DoglegParams());
  }
  isam2_params.relinearizeSkip = static_cast<int>(params.isam2_relinearize_skip);
  isam2_params.setRelinearizeThreshold(params.isam2_relinearize_thresh);
  if (params.enable_fixed_lag_smoother) {
    smoother.reset(new FixedLagSmootherExt(params.smoother_lag, isam2_params));
  }

#ifdef GTSAM_USE_TBB
  tbb_task_arena = std::make_shared<tbb::task_arena>(1);
#endif

  Callbacks::request_to_compute_covariances.add([this]() {
    logger->debug("request to compute marginal covs");
    this->params.compute_covs = true;
  });
}

OdometryEstimationCT::~OdometryEstimationCT() {}

gtsam::Pose3 OdometryEstimationCT::estimate_pose(gtsam::Key key) const {
  if (params.enable_fixed_lag_smoother) {
    return smoother->calculateEstimate<gtsam::Pose3>(key);
  }
  return scan_poses_.at<gtsam::Pose3>(key);
}

EstimationFrame::ConstPtr OdometryEstimationCT::insert_frame(const PreprocessedFrame::Ptr& raw_frame, std::vector<EstimationFrame::ConstPtr>& marginalized_frames) {
  Callbacks::on_insert_frame(raw_frame);

  const int current = frames.size();
  const int last = current - 1;

  // Create a frame to hold the current estimation state
  EstimationFrame::Ptr new_frame(new EstimationFrame);
  new_frame->id = current;
  new_frame->stamp = raw_frame->stamp;
  new_frame->T_lidar_imu.setIdentity();
  new_frame->v_world_imu.setZero();
  new_frame->imu_bias.setZero();
  new_frame->raw_frame = raw_frame;

  gtsam_points::PointCloudCPU::Ptr frame_cpu(new gtsam_points::PointCloudCPU(raw_frame->points));
  frame_cpu->add_times(raw_frame->times);

  covariance_estimation->estimate(raw_frame->points, raw_frame->neighbors, frame_cpu->normals_storage, frame_cpu->covs_storage);
  frame_cpu->normals = frame_cpu->normals_storage.data();
  frame_cpu->covs = frame_cpu->covs_storage.data();

  new_frame->frame = frame_cpu;
  new_frame->frame_id = FrameID::LIDAR;

  // New values and factors to be inserted into the smoother
  // note: X(i) and Y(i) respectively represent the sensor poses at the scan beginning and ending of i-th frame
  gtsam::FixedLagSmootherKeyTimestampMap new_stamps;
  gtsam::Values new_values;
  gtsam::NonlinearFactorGraph new_factors;

  if (current == 0) {
    new_frame->set_T_world_sensor(FrameID::LIDAR, Eigen::Isometry3d::Identity());
    new_frame->T_world_lidar_end = Eigen::Isometry3d::Identity();

    new_stamps[X(0)] = new_frame->stamp;
    new_stamps[Y(0)] = new_frame->stamp;
    new_values.insert(X(0), gtsam::Pose3());
    new_values.insert(Y(0), gtsam::Pose3());
    new_factors.emplace_shared<gtsam::PriorFactor<gtsam::Pose3>>(X(0), gtsam::Pose3(), gtsam::noiseModel::Isotropic::Precision(6, 1e6));
    new_factors.emplace_shared<gtsam::BetweenFactor<gtsam::Pose3>>(X(0), Y(0), gtsam::Pose3(), gtsam::noiseModel::Isotropic::Precision(6, 1e6));
  } else {
    // Estimate sensor velocity from last sensor states
    gtsam::Vector6 last_twist = gtsam::Vector6::Zero();
    if (current >= 2) {
      if (!frames[last] || !frames[last - 1]) {
        logger->warn(
          "neither frames[last]={} nor frames[last - 1]={} is released!!",
          static_cast<const void*>(frames[last].get()),
          static_cast<const void*>(frames[last - 1].get()));
        logger->warn("there might be a large time gap between point cloud frames");
      } else {
        const double delta_time = (frames[last]->stamp + frames[last]->frame->times[frames[last]->frame->size() - 1]) - frames[last - 1]->stamp;
        const gtsam::Pose3 delta_pose = estimate_pose(X(last - 1)).inverse() * estimate_pose(Y(last));
        last_twist = 0.85 * gtsam::Pose3::Logmap(delta_pose) / delta_time;
      }
    }

    const auto& last_frame = frames[last];
    const double last_time_begin = last_frame->stamp + last_frame->frame->times[0];
    const double last_time_end = last_frame->stamp + last_frame->frame->times[last_frame->frame->size() - 1];
    const auto last_T_world_lidar_begin_ = estimate_pose(X(last));
    const auto last_T_world_lidar_end_ = estimate_pose(Y(last));
    const auto last_T_world_lidar_begin = gtsam::Pose3(last_T_world_lidar_begin_.rotation().normalized(), last_T_world_lidar_begin_.translation());
    const auto last_T_world_lidar_end = gtsam::Pose3(last_T_world_lidar_end_.rotation().normalized(), last_T_world_lidar_end_.translation());

    // Initial guess
    const double current_time_begin = new_frame->stamp + new_frame->frame->times[0];
    const double current_time_end = new_frame->stamp + new_frame->frame->times[new_frame->frame->size() - 1];
    const gtsam::Pose3 predicted_T_world_lidar_begin = last_T_world_lidar_end * gtsam::Pose3::Expmap(last_twist * (current_time_begin - last_time_end));
    const gtsam::Pose3 predicted_T_world_lidar_end = predicted_T_world_lidar_begin * gtsam::Pose3::Expmap(last_twist * (current_time_end - current_time_begin));

    gtsam::Values values;
    values.insert(X(current), gtsam::Pose3(predicted_T_world_lidar_begin));
    values.insert(Y(current), gtsam::Pose3(predicted_T_world_lidar_end));

    // Create CT-GICP factor
    gtsam::NonlinearFactorGraph graph;
    auto factor =
      gtsam::make_shared<gtsam_points::IntegratedCT_GICPFactor_<gtsam_points::iVox, gtsam_points::PointCloud>>(X(current), Y(current), target_ivox, new_frame->frame, target_ivox);
    factor->set_num_threads(params.num_threads);
    factor->set_max_correspondence_distance(params.max_correspondence_distance);
    {
      const double t_end = raw_frame->full_scan_duration > 0.0 ? raw_frame->full_scan_duration : raw_frame->times.back();
      factor->set_time_span(0.0, t_end);
    }
    graph.add(factor);

    // Location consistency & constant velocity constraints
    graph.emplace_shared<gtsam::PriorFactor<gtsam::Pose3>>(X(current), last_T_world_lidar_end, gtsam::noiseModel::Isotropic::Precision(6, params.location_consistency_inf_scale));
    graph
      .emplace_shared<gtsam::BetweenFactor<gtsam::Pose3>>(X(current), Y(current), gtsam::Pose3(), gtsam::noiseModel::Isotropic::Precision(6, params.constant_velocity_inf_scale));

    // LM configuration
    gtsam_points::LevenbergMarquardtExtParams lm_params;
    lm_params.setlambdaInitial(1e-10);
    lm_params.setAbsoluteErrorTol(1e-2);
    lm_params.setMaxIterations(params.lm_max_iterations);
    // lm_params.set_verbose();

    try {
#ifdef GTSAM_USE_TBB
      auto arena = static_cast<tbb::task_arena*>(tbb_task_arena.get());
      arena->execute([&] {
#endif
        values = gtsam_points::LevenbergMarquardtOptimizerExt(graph, values, lm_params).optimize();
#ifdef GTSAM_USE_TBB
      });
#endif

    } catch (std::exception& e) {
      logger->error("an exception was caught during odometry estimation");
      logger->error("{}", e.what());
    }

    const gtsam::Pose3 T_world_lidar_begin = values.at<gtsam::Pose3>(X(current));
    const gtsam::Pose3 T_world_lidar_end = values.at<gtsam::Pose3>(Y(current));

    new_frame->v_world_imu = (T_world_lidar_begin.translation() - last_T_world_lidar_begin.translation()) / (current_time_begin - last_time_begin);
    new_frame->set_T_world_sensor(FrameID::LIDAR, Eigen::Isometry3d(values.at<gtsam::Pose3>(X(current)).matrix()));
    new_frame->T_world_lidar_end = Eigen::Isometry3d(values.at<gtsam::Pose3>(Y(current)).matrix());

    // Deskew the input points and covs
    auto deskewed_points = factor->deskewed_source_points(values, true);
    auto deskewed_covs = covariance_estimation->estimate(deskewed_points, raw_frame->neighbors);
    for (int i = 0; i < deskewed_points.size(); i++) {
      new_frame->frame->points[i] = deskewed_points[i];
      new_frame->frame->covs[i] = deskewed_covs[i];
    }
    Callbacks::on_new_frame(new_frame);

    new_stamps[X(current)] = new_frame->stamp;
    new_stamps[Y(current)] = new_frame->stamp;
    new_values.insert(X(current), T_world_lidar_begin);
    new_values.insert(Y(current), T_world_lidar_end);

    new_factors.emplace_shared<gtsam::BetweenFactor<gtsam::Pose3>>(
      Y(last),
      X(current),
      last_T_world_lidar_end.inverse() * T_world_lidar_begin,
      gtsam::noiseModel::Isotropic::Precision(6, 1e3));
    new_factors.emplace_shared<gtsam::BetweenFactor<gtsam::Pose3>>(
      X(current),
      Y(current),
      T_world_lidar_begin.inverse() * T_world_lidar_end,
      gtsam::noiseModel::Isotropic::Precision(6, 1e3));
    new_factors.emplace_shared<gtsam::PriorFactor<gtsam::Pose3>>(X(current), T_world_lidar_begin, gtsam::noiseModel::Isotropic::Precision(6, 1e3));
    new_factors.emplace_shared<gtsam::PriorFactor<gtsam::Pose3>>(Y(current), T_world_lidar_end, gtsam::noiseModel::Isotropic::Precision(6, 1e3));
  }

  Callbacks::on_new_frame(new_frame);
  frames.push_back(new_frame);

  // Transform points into the global coordinate and insert them into the iVox
  auto transformed = gtsam_points::PointCloudCPU::clone(*new_frame->frame);
  for (int i = 0; i < transformed->size(); i++) {
    transformed->points[i] = new_frame->T_world_sensor() * new_frame->frame->points[i];
    transformed->covs[i] = new_frame->T_world_sensor().matrix() * new_frame->frame->covs[i] * new_frame->T_world_sensor().matrix().transpose();
  }
  target_ivox->insert(*transformed);

  if (params.enable_fixed_lag_smoother) {
    Callbacks::on_smoother_update(*smoother, new_factors, new_values, new_stamps);
#ifdef GTSAM_USE_TBB
    auto arena = static_cast<tbb::task_arena*>(tbb_task_arena.get());
    arena->execute([&] {
#endif
      smoother->update(new_factors, new_values, new_stamps);
#ifdef GTSAM_USE_TBB
    });
#endif
    Callbacks::on_smoother_update_finish(*smoother);
  } else {
    for (const auto& key_value : new_values) {
      scan_poses_.insert(key_value.key, key_value.value);
    }
  }

  // Find out marginalized frames
  while (marginalized_cursor < current) {
    double span = frames[current]->stamp - frames[marginalized_cursor]->stamp;
    if (span < params.smoother_lag - 0.1) {
      break;
    }

    marginalized_frames.push_back(frames[marginalized_cursor]);
    frames[marginalized_cursor].reset();
    marginalized_cursor++;
  }
  Callbacks::on_marginalized_frames(marginalized_frames);

  // Update estimation frames
  for (int i = marginalized_cursor; i < frames.size(); i++) {
    try {
      Eigen::Isometry3d T_world_lidar = Eigen::Isometry3d(estimate_pose(X(i)).matrix());
      frames[i]->set_T_world_sensor(FrameID::LIDAR, T_world_lidar);
      frames[i]->T_world_lidar_end = Eigen::Isometry3d(estimate_pose(Y(i)).matrix());
    } catch (std::out_of_range& e) {
      logger->error("caught {}", e.what());
      logger->error("current={}", current);
      logger->error("marginalized_cursor={}", marginalized_cursor);
      Callbacks::on_smoother_corruption(frames[current]->stamp);
      break;
    }
  }

  if (params.compute_covs && params.enable_fixed_lag_smoother) {
    logger->debug("computing marginal covariances of the latest frame");
    new_frame->cov_computed = true;
    new_frame->pose_cov = smoother->marginalCovariance(X(current));
  }

  std::vector<EstimationFrame::ConstPtr> active_frames(frames.begin() + marginalized_cursor, frames.end());
  Callbacks::on_update_new_frame(active_frames.back());
  Callbacks::on_update_frames(active_frames);

  // Update target point cloud (just for visualization)
  // This is not necessary for mapping and can safely be removed.
  if (frames.size() % 100 == 0) {
    EstimationFrame::Ptr frame(new EstimationFrame);
    frame->id = current;
    frame->stamp = new_frame->stamp;

    frame->T_lidar_imu.setIdentity();
    frame->T_world_lidar.setIdentity();
    frame->T_world_imu.setIdentity();

    frame->v_world_imu.setZero();
    frame->imu_bias.setZero();

    frame->frame_id = FrameID::LIDAR;
    frame->frame = std::make_shared<gtsam_points::PointCloudCPU>(target_ivox->voxel_points());

    std::vector<EstimationFrame::ConstPtr> keyframes = {frame};
    Callbacks::on_update_keyframes(keyframes);

    if (target_ivox_frame) {
      std::vector<EstimationFrame::ConstPtr> marginalized_keyframes = {target_ivox_frame};
      Callbacks::on_marginalized_keyframes(marginalized_keyframes);
    }

    target_ivox_frame = frame;
  }

  return new_frame;
}

}  // namespace glim
