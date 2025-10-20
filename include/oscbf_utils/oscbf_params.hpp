#pragma once
#include <Eigen/Dense>

// We reuse your damper fields for smooth joint-limit handling near bounds.
struct OSCBFParams {
  // Task-consistent objective weights (diagonal). If empty → identity.
  Eigen::VectorXd Wj_diag;   // size: nq (joint-space deviation weight)
  Eigen::VectorXd Wo_diag;   // size: 6  (op-space deviation weight)

  // CBF scalars
  double eps         = 1e-2;   // µ tolerance ε (keeps away from singularity)
  double alpha_gain  = 10.0;   // α(h) = alpha_gain * h
  double slack_rho   = 1e6;    // large penalty on slack t
  double slack_max   = 1e3;    // upper bound on slack, >=0

  // Velocity bounds fallback (if not supplied by joint_limits col 2)
  Eigen::VectorXd qd_min;  // size nq
  Eigen::VectorXd qd_max;  // size nq

  // Velocity damper near joint limits
  bool   use_joint_limit_damper = true;
  double eta   = 0.8;
  double rho_i = 30.0 * M_PI/180.0; // inner zone (rad)
  double rho_s = 5.0  * M_PI/180.0; // stop dist (rad)

  // Optional scaling between J and N objectives (if magnitudes differ).
  double objective_scale = 1.0;
};
