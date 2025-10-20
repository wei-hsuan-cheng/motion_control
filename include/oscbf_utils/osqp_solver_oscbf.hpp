#pragma once
#include "qp_solver/iqpsolver.hpp"

// OSQP adapter specialized for OSCBF QP structure.
// Assumes decision x = [qdot(n); t(1)],
// inequalities stack as: [ CBF(1); damper(2n) ], and bounds on variables.
class OsqpSolverOscbf : public IQPSolver {
public:
  OsqpSolverOscbf();
  ~OsqpSolverOscbf() override;

  QPResult solve(const QPProblem& p) override;

  // Optional: set OSQP parameters
  void setRho(double rho);
  void setEpsAbs(double eps);
  void setEpsRel(double eps);
  void setMaxIter(int iters);
  void setVerbose(bool v);

private:
  struct Impl;
  Impl* impl_;
};

