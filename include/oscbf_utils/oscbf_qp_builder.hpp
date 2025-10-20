#pragma once
#include <Eigen/Dense>
#include "qp_solver/iqpsolver.hpp"
#include "oscbf_utils/oscbf_params.hpp"

// Kinematic OSCBF (singularity avoidance) minimal QP builder.
// Decision x = [ qdot(n); t(1) ]
// Objective: ||Wj N(q)(qdot - qdot_nom)||^2 + ||Wo J(q)(qdot - qdot_nom)||^2 + rho * t
// CBF: (∇µ)^T qdot  >= -α(h(q)) - t    with  h(q)=µ(q)-ε,  t∈[0,slack_max]
class OSCBFQPBuilder {
public:
  explicit OSCBFQPBuilder(const OSCBFParams& P) : P_(P) {}

  QPProblem build(const Eigen::MatrixXd& J,
                  const Eigen::MatrixXd& N,           // null projector (I - J^+ J)
                  const Eigen::VectorXd& qdot_nom,    // nominal joint vel
                  const Eigen::VectorXd& grad_mu,     // ∇µ(q)  (size n)
                  double mu                           // µ(q)
                 ) const
  {
    const int n  = (int)J.cols();
    const int nv = n + 1; // qdot + slack t

    QPProblem prob;
    // ----- Objective matrices -----
    // PQP = (N^T Wj^2 N + J^T Wo^2 J) on the qdot block; slack has 0 quad cost.
    Eigen::VectorXd Wj = (P_.Wj_diag.size()==n) ? P_.Wj_diag : Eigen::VectorXd::Ones(n);
    Eigen::VectorXd Wo = (P_.Wo_diag.size()==6) ? P_.Wo_diag : Eigen::VectorXd::Ones(6);
    Eigen::MatrixXd WjM = Wj.asDiagonal();
    Eigen::MatrixXd WoM = Wo.asDiagonal();

    Eigen::MatrixXd PQP = (N.transpose() * WjM * WjM * N)
                        + (J.transpose() * WoM * WoM * J);
    PQP *= P_.objective_scale;

    prob.Q.setZero(nv, nv);
    prob.Q.topLeftCorner(n, n) = PQP;

    // Linear term q_QP = - qdot_nom^T * PQP  (slack linear coeff = ρ)
    prob.c.setZero(nv);
    prob.c.head(n) = -PQP * qdot_nom;
    prob.c(n) = P_.slack_rho;

    // ----- Equality: none -----
    prob.Aeq.resize(0, nv);
    prob.beq.resize(0);

    // ----- Inequalities: only CBF -----
    // CBF row on x=[qdot; t]:  [ grad_mu^T   ,  1 ] * [qdot; t]  >= -α(µ-ε)
    // We encode as <=: -grad^T qdot - t <= α(h)
    const int mi = 1;             // 1 CBF row
    prob.Aineq.setZero(mi, nv);
    prob.bineq.setZero(mi);

    // CBF (row 0): -grad^T qdot - 1*t <= α(h)  (we keep <= form)
    // α(h)=alpha_gain*(µ-ε)
    const double h  = mu - P_.eps;
    const double rhs = P_.alpha_gain * h;
    prob.Aineq.row(0).head(n) = -grad_mu.transpose(); // -grad^T qdot
    prob.Aineq(0, n) = -1.0;                          // -t
    prob.bineq(0) = rhs;

    // ----- Bounds on variables -----
    prob.lb = Eigen::VectorXd::Constant(nv, -std::numeric_limits<double>::infinity());
    prob.ub = Eigen::VectorXd::Constant(nv,  std::numeric_limits<double>::infinity());

    // Slack bounds
    prob.lb(n) = 0.0;
    prob.ub(n) = P_.slack_max;

    return prob;
  }

private:
  OSCBFParams P_;
};
