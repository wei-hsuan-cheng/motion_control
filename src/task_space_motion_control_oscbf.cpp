#include <rclcpp/rclcpp.hpp>
#include "robot_math_utils/robot_math_utils_v1_17.hpp"
#include "robot_kinematics_utils/robot_kinematics_utils_v1_0.hpp"

#include <geometry_msgs/msg/pose_stamped.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <Eigen/Dense>
#include <unordered_map>
#include <deque>
#include <chrono>
#include <limits>
#include <memory>
#include <algorithm>
#include <cmath>

// Existing QP infra
#include "qp_solver/iqpsolver.hpp"
#include "oscbf_utils/osqp_solver_oscbf.hpp"
#include "oscbf_utils/oscbf_params.hpp"
#include "oscbf_utils/oscbf_qp_builder.hpp"

using RM = RMUtils;
using std::placeholders::_1;
using HighResClock = std::chrono::high_resolution_clock;

class TaskSpaceMotionControlOSCBF : public rclcpp::Node {
public:
  TaskSpaceMotionControlOSCBF()
  : rclcpp::Node("task_space_motion_control_oscbf")
  {
    loadParams();
    initRobotConfig();
    initState();
    initOSCBF();
    initROS();
    RCLCPP_INFO(get_logger(), "OSCBF (singularity) node ready with %d joints @ %.1f Hz", n_, fs_);
  }

private:
  // ---------- Params & robot ----------
  void loadParams(){
    declare_parameter<std::string>("robot_name", "");
    declare_parameter<std::string>("base_link", "");
    declare_parameter<std::string>("ee_link", "");
    declare_parameter<std::string>("screw_representation", "body");
    declare_parameter<std::vector<std::string>>("joint_names", {});
    declare_parameter<int>("num_joints", 0);
    declare_parameter<std::vector<double>>("joint_limits_lower",   {});
    declare_parameter<std::vector<double>>("joint_limits_upper",   {});
    declare_parameter<std::vector<double>>("joint_limits_velocity",{});
    declare_parameter<std::vector<double>>("joint_limits_effort",  {});
    declare_parameter<std::vector<double>>("M_position", {});
    declare_parameter<std::vector<double>>("M_quaternion_wxyz", {});
    declare_parameter<double>("fs", 100.0);

    // OSCBF params (override at runtime if you want)
    declare_parameter<double>("oscbf.eps",        1e-3);
    declare_parameter<double>("oscbf.alpha_gain", 1e1);
    declare_parameter<double>("oscbf.slack_rho",  1e6);
    declare_parameter<double>("oscbf.slack_max",  1e3);
    declare_parameter<double>("oscbf.obj_scale",  1.0);
    // Uniform diagonal weights (scalars)
    declare_parameter<double>("oscbf.wj_scalar",  1.0);
    declare_parameter<double>("oscbf.wo_scalar",  1.0);
    // Minimal OSCBF: no limit toggles or dynamic Wo scaling
  }

  void initRobotConfig(){
    get_parameter("robot_name", robot_name_);
    get_parameter("base_link", base_link_);
    get_parameter("ee_link", ee_link_);
    get_parameter("screw_representation", rep_);
    get_parameter("joint_names", joint_names_);
    get_parameter("num_joints", num_joints_param_);
    get_parameter("M_position", M_pos_);
    get_parameter("M_quaternion_wxyz", M_qwxyz_);
    get_parameter("fs", fs_);

    if (joint_names_.empty()) throw std::runtime_error("joint_names empty");
    n_ = (int)joint_names_.size();

    std::vector<double> jl_lower, jl_upper, jl_vel, jl_eff;
    get_parameter("joint_limits_lower", jl_lower);
    get_parameter("joint_limits_upper", jl_upper);
    get_parameter("joint_limits_velocity", jl_vel);
    get_parameter("joint_limits_effort", jl_eff);
    auto ensure=[&](std::vector<double>& v){ if((int)v.size()!=n_) v.resize(n_,0.0); };
    ensure(jl_lower); ensure(jl_upper); ensure(jl_vel); ensure(jl_eff);

    joint_limits_.resize(n_,4);
    for(int i=0;i<n_;++i){
      joint_limits_(i,0)=jl_lower[i];
      joint_limits_(i,1)=jl_upper[i];
      joint_limits_(i,2)=std::abs(jl_vel[i]);
      joint_limits_(i,3)=jl_eff[i];
    }

    if (M_pos_.size()!=3 || M_qwxyz_.size()!=4) throw std::runtime_error("Bad M sizes");
    M_.pos  = Eigen::Vector3d(M_pos_[0], M_pos_[1], M_pos_[2]);
    M_.quat = Eigen::Quaterniond(M_qwxyz_[0], M_qwxyz_[1], M_qwxyz_[2], M_qwxyz_[3]);
    M_.quat.normalize();

    S_.resize(6,n_);
    for (int j=0;j<n_;++j){
      const std::string pname="screw_list."+joint_names_[j];
      declare_parameter<std::vector<double>>(pname,{});
      std::vector<double> a; get_parameter(pname,a);
      if(a.size()!=6) throw std::runtime_error("Bad screw param: "+pname);
      S_.col(j) << a[0],a[1],a[2],a[3],a[4],a[5];
    }
    screws_ = ScrewList(S_, M_);
    screws_.setMeta(robot_name_, ScrewList::ParseRep(rep_), joint_names_,
                    base_link_.empty()?"base_link":base_link_,
                    ee_link_.empty()  ?"ee_link"  :ee_link_, joint_limits_);

    if (fs_<=0.0) throw std::runtime_error("fs must be >0");
    Ts_=1.0/fs_;
  }

  void initState(){
    rk_ = std::make_unique<RKUtils>(screws_);
    q_.setZero(n_); qd_.setZero(n_); qd_cmd_.setZero(n_);
    rk_->UpdateRobotState(q_, qd_);
    pos_quat_cmd_ = rk_->pose();

    // simple P gains for op-space tracking (same style as your TSMC)
    double pos_mult = 5.0*std::pow(10.0,2.0);
    double so3_mult = 5.0*std::pow(10.0,2.0);
    Eigen::Vector3d kp_pos(1,1,1), kp_so3(1,1,1);
    Kp_.setIdentity();
    Kp_.topLeftCorner(3,3) = (pos_mult*kp_pos).asDiagonal();
    Kp_.bottomRightCorner(3,3)=(so3_mult*kp_so3).asDiagonal();

    last_log_ = std::chrono::steady_clock::now();
  }

  void initOSCBF(){
    // OSQP solver
    solver_ = std::make_unique<OsqpSolverOscbf>();
    static_cast<OsqpSolverOscbf*>(solver_.get())->setVerbose(false);
    static_cast<OsqpSolverOscbf*>(solver_.get())->setMaxIter(4000);

    // params
    double rho_i_deg, rho_s_deg;
    get_parameter("oscbf.eps",        P_.eps);
    get_parameter("oscbf.alpha_gain", P_.alpha_gain);
    get_parameter("oscbf.slack_rho",  P_.slack_rho);
    get_parameter("oscbf.slack_max",  P_.slack_max);
    get_parameter("oscbf.obj_scale",  P_.objective_scale);
    get_parameter("oscbf.wj_scalar", wj_scalar_base_);
    get_parameter("oscbf.wo_scalar", wo_scalar_base_);
    P_.Wj_diag = Eigen::VectorXd::Constant(n_, wj_scalar_base_);
    P_.Wo_diag = Eigen::VectorXd::Constant(6,  wo_scalar_base_);

    // log params
    RCLCPP_INFO(get_logger(), "[OSCBF params]:\n eps=%.3e\n alpha_gain=%.3e\n slack_rho=%.3e\n slack_max=%.3e\n obj_scale=%.3e\n Wj_scalar=%.3e\n Wo_scalar=%.3e",
                P_.eps, P_.alpha_gain, P_.slack_rho, P_.slack_max,
                P_.objective_scale,
                wj_scalar_base_, wo_scalar_base_);

  }

  void initROS(){
    sub_pose_cmd_ = create_subscription<geometry_msgs::msg::PoseStamped>(
      "/pose_command", rclcpp::SensorDataQoS(), std::bind(&TaskSpaceMotionControlOSCBF::onPose, this, _1));
    sub_js_ = create_subscription<sensor_msgs::msg::JointState>(
      "/joint_states", rclcpp::SensorDataQoS(), std::bind(&TaskSpaceMotionControlOSCBF::onJS, this, _1));
    pub_qd_ = create_publisher<sensor_msgs::msg::JointState>("/joint_velocity_command", rclcpp::SensorDataQoS());

    timer_ = create_wall_timer(
      std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::duration<double>(Ts_)),
      std::bind(&TaskSpaceMotionControlOSCBF::onTimer, this));
  }

  // Helpers
  // Per-call timing helper
  static inline double elapsedUs(const HighResClock::time_point& t0,
                                const HighResClock::time_point& t1) {
      return std::chrono::duration<double, std::micro>(t1 - t0).count();
  }

  // ---------- Callbacks ----------
  void onPose(const geometry_msgs::msg::PoseStamped& msg){
    Eigen::Vector3d p(msg.pose.position.x, msg.pose.position.y, msg.pose.position.z);
    Eigen::Quaterniond q(msg.pose.orientation.w, msg.pose.orientation.x,
                         msg.pose.orientation.y, msg.pose.orientation.z);
    pos_quat_cmd_ = PosQuat(p,q);
    have_cmd_=true;
  }

  void onJS(const sensor_msgs::msg::JointState& msg){
    if (msg.name.empty()) return;
    std::unordered_map<std::string, std::size_t> idx;
    for (std::size_t k=0;k<msg.name.size();++k) idx[msg.name[k]]=k;

    for (int i=0;i<n_;++i){
      auto it=idx.find(joint_names_[i]);
      if (it==idx.end()) continue;
      size_t k=it->second;
      if (k<msg.position.size()) q_(i)=msg.position[k];
      if (k<msg.velocity.size()) qd_(i)=msg.velocity[k];
    }
    rk_->UpdateRobotState(q_, qd_);
    have_js_=true;
  }

  // ---------- Control loop ----------
  void onTimer(){
    if (!have_js_ || !have_cmd_){
      if (logTick(1.0)) RCLCPP_WARN(get_logger(),"Waiting for /joint_states and /pose_command ...");
      return;
    }

    // 1) Operational-space P control to build nominal ν
    PosQuat rel = RM::PosQuats2RelativePosQuat(rk_->pose(), pos_quat_cmd_);
    Eigen::Matrix<double,6,1> e = RM::PosQuat2Posso3(rel);
    Eigen::Matrix<double,6,1> nu = RM::KpPosso3(e, Kp_, /*target_reached*/false);

    auto sat = [](const Eigen::Vector3d& v, double lim) {
    double n=v.norm(); return (n>lim && n>1e-12) ? (v*(lim/n)).eval() : v;
    };
    double max_lin = 1.0;              // [m]
    double max_ang = 0.5 * M_PI;              // [rad/s]
    nu.head<3>() = sat(nu.head<3>(), max_lin);
    nu.tail<3>() = sat(nu.tail<3>(), max_ang);

    auto t0 = HighResClock::now();
    // 2) Nominal qdot: q̇_nom = J^+ ν + N q̇_null; here q̇_null= (1/λ) (I_n - J^+J)∇μ
    double lambda_dls = 5e-2;
    // double lambda_dls = 0.0;
    Eigen::MatrixXd Jpinv = rk_->JacobPinvDLS(rk_->jacob(), lambda_dls);

    double lambda = 1.0e0; // Secondary objective gain
    Eigen::MatrixXd N = Eigen::MatrixXd::Identity(n_,n_) - Jpinv * rk_->jacob();
    Eigen::VectorXd qdot_nom = Jpinv * nu + (1 / lambda) * N * rk_->manipulability_gradient(); // velocity components for primary + secondary objetives
    // Eigen::VectorXd qdot_nom = Jpinv * nu; // velocity components for primary objetive only

    // 3) Build OSCBF QP (minimal)
    // Compute μ and ∇μ once and reuse
    const double mu_now = rk_->manipulability();
    const Eigen::VectorXd grad_now = rk_->manipulability_gradient();
    P_.Wo_diag = Eigen::VectorXd::Constant(6, wo_scalar_base_);

    OSCBFQPBuilder builder(P_);
    QPProblem prob = builder.build(rk_->jacob(),
                                   N,
                                   qdot_nom,
                                   grad_now,
                                   mu_now);

    // 4) Solve
    QPResult sol = solver_->solve(prob);
    if (!sol.ok || sol.x.size() != n_+1) {
      // Safety fallback: stick to nominal within bounds
      qd_cmd_ = qdot_nom.cwiseMax(prob.lb.head(n_)).cwiseMin(prob.ub.head(n_));
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000, "\n========================\n[OSCBF] QP failed → using bounded nominal.\n========================");
    } else {
      qd_cmd_ = sol.x.head(n_);
    }

    auto t1 = HighResClock::now();
    double us = elapsedUs(t0, t1);
    // RCLCPP_INFO(get_logger(), "[OSCBF] Solve time/rate: %.3f [us] / %.1f [Hz]", us, 1e6/us);


    // 5) Publish
    sensor_msgs::msg::JointState js;
    js.header.stamp = now();
    js.name = joint_names_;
    js.velocity.assign(qd_cmd_.data(), qd_cmd_.data()+qd_cmd_.size());
    pub_qd_->publish(js);

    // log
    if (logTick(0.5)) {
      const double h_now  = mu_now - P_.eps;
      const double t_val = (sol.ok && sol.x.size()==n_+1) ? sol.x(n_) : 0.0;
      const double lhs   = grad_now.dot(qd_cmd_) + t_val;           // native CBF LHS (>= rhs)
      const double rhs   = -P_.alpha_gain * h_now;                  // native CBF RHS
      const double v_n = (rk_->jacob() * qd_cmd_).head<3>().norm();
      const double omg_n = (rk_->jacob() * qd_cmd_).tail<3>().norm();
      RCLCPP_INFO(get_logger(), "[OSCBF] μ=%.3e, h=%.3e, |qd|=%.3f, |v|=%.3f, |omg|=%.3f, lhs=%.3e, rhs=%.3e, (lhs - rhs) / |rhs| =%.3e",
                  mu_now, h_now, qd_cmd_.norm(), v_n, omg_n, lhs, rhs, (lhs - rhs) / std::abs(rhs));

      // Print the desired twist v.s. optimized twist and twist error
      Eigen::VectorXd twist_opt = rk_->jacob() * qd_cmd_;
      Eigen::VectorXd twist_err = nu - twist_opt;
      RCLCPP_INFO(get_logger(), "[OSCBF] desired twist: [%.3f, %.3f, %.3f, %.3f, %.3f, %.3f]^T",
                  nu(0), nu(1), nu(2), nu(3), nu(4), nu(5));
      RCLCPP_INFO(get_logger(), "[OSCBF] optimized twist: [%.3f, %.3f, %.3f, %.3f, %.3f, %.3f]^T",
                  twist_opt(0), twist_opt(1), twist_opt(2),
                  twist_opt(3), twist_opt(4), twist_opt(5));
      RCLCPP_INFO(get_logger(), "[OSCBF] twist error:   [%.3f, %.3f, %.3f, %.3f, %.3f, %.3f]^T",
                  twist_err(0), twist_err(1), twist_err(2),
                  twist_err(3), twist_err(4), twist_err(5));

    }
  }

  bool logTick(double s){
    auto now = std::chrono::steady_clock::now();
    if (std::chrono::duration<double>(now - last_log_).count() >= s) { last_log_=now; return true; }
    return false;
  }

private:
  // config
  std::string robot_name_, base_link_, ee_link_, rep_;
  std::vector<std::string> joint_names_;
  int n_{0}, num_joints_param_{0};
  double fs_{100.0}, Ts_{0.01};

  // kinematics & state
  ScrewList screws_; PosQuat M_;
  std::vector<double> M_pos_, M_qwxyz_;
  Eigen::MatrixXd S_, joint_limits_;
  std::unique_ptr<RKUtils> rk_;
  Eigen::VectorXd q_, qd_, qd_cmd_;
  PosQuat pos_quat_cmd_;
  Eigen::Matrix<double,6,6> Kp_;

  // oscbf
  std::unique_ptr<IQPSolver> solver_;
  OSCBFParams P_;
  // uniform weight scalars (diagonal)
  double wj_scalar_base_{1.0};
  double wo_scalar_base_{1.0};

  // ros
  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr sub_pose_cmd_;
  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr sub_js_;
  rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr pub_qd_;
  rclcpp::TimerBase::SharedPtr timer_;

  // book-keeping
  bool have_js_{false}, have_cmd_{false};
  std::chrono::steady_clock::time_point last_log_;
};

int main(int argc, char** argv){
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<TaskSpaceMotionControlOSCBF>());
  rclcpp::shutdown();
  return 0;
}
