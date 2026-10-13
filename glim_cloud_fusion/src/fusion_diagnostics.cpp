#include <glim_cloud_fusion/fusion.hpp>

#include <iomanip>

namespace glim_cloud_fusion {

void print_fusion_integrator_diagnostics(std::ostream& out, int scan_index, const FusionIntegratorDiagnostics& d) {
  out << "[integrator-diag] scan=" << scan_index << (d.is_seed_scan ? " (seed)" : "") << "\n"
      << "  raw_points=" << d.num_raw_points << " subsample=" << d.num_subsample_points
      << " target_map_pts=" << d.target_map_points << " output_voxels=" << d.output_voxel_count << "\n"
      << std::fixed << std::setprecision(6) << "  time_s=[" << d.time_min_s << ", " << d.time_max_s << "] alpha=["
      << d.alpha_min << ", " << d.alpha_max << "]\n"
      << "  lm_error initial=" << d.lm_error_initial << " final=" << d.lm_error_final << "\n"
      << std::setprecision(4) << "  scan_motion trans_m=" << d.scan_begin_end_trans_m
      << " rot_deg=" << d.scan_begin_end_rot_deg << "\n"
      << "  odom_end trans_m=" << d.odom_end_trans_m << " rot_deg=" << d.odom_end_rot_deg << "\n";
}

}  // namespace glim_cloud_fusion
