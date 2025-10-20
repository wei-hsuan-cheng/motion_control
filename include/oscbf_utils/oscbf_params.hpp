#pragma once
#include <Eigen/Dense>

struct OSCBFParams {
  // Task-consistent objective weights (diagonal). If empty → identity.
  Eigen::VectorXd Wj_diag;   // size: nq (joint-space deviation weight)
  Eigen::VectorXd Wo_diag;   // size: 6  (op-space deviation weight)

  // CBF scalars
  double eps         = 1e-2;   // µ tolerance ε (keeps away from singularity)
  double alpha_gain  = 10.0;   // α(h) = alpha_gain * h
  double slack_rho   = 1e6;    // large penalty on slack t
  double slack_max   = 1e3;    // upper bound on slack, >=0

  // Optional scaling between J and N objectives (if magnitudes differ).
  double objective_scale = 1.0;
};
