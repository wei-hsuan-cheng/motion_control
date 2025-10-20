cmake_minimum_required(VERSION 3.8)
project(motion_control)

# Compiler settings
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Set C++ standard
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

# Suppress warnings for Eigen
add_compile_options(-Wno-class-memaccess)

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_action REQUIRED)  # <-- for using actions in C++
find_package(rclcpp_components REQUIRED)  # <-- for using actions in C++
find_package(Eigen3 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(osqp REQUIRED)


# Dependencies list
set(dependencies
    ament_cmake
    rclcpp
    rclcpp_action
    rclcpp_components
    Eigen3
    tf2_ros
    sensor_msgs
    visualization_msgs
    geometry_msgs
)

# For ROS 2 action
ament_export_dependencies(rosidl_default_runtime)


# Include directories
include_directories(
  include
  ${EIGEN3_INCLUDE_DIRS}
)

# Find OSQP target
set(OSQP_TARGET "")
# Prefer static to avoid runtime loader issues with ~/.local
if(TARGET osqp::osqpstatic)
  set(OSQP_TARGET osqp::osqpstatic)
elseif(TARGET osqp::osqp)
  set(OSQP_TARGET osqp::osqp)
elseif(TARGET OSQP::osqp)
  set(OSQP_TARGET OSQP::osqp)
else()
  # Fall back to 'osqp' if no imported target is defined
  set(OSQP_TARGET osqp)
endif()

# Source files
set(ROBOT_JOINT_DYNAMICS_SOURCES src/robot_joint_dynamics.cpp)
set(TASK_SPACE_MOTION_CONTROL_SOURCES src/task_space_motion_control.cpp)
set(MOTION_REFERENCE_GENERATOR_SOURCES src/motion_reference_generator.cpp)
set(TASK_SPACE_OSCBF_SOURCES src/task_space_motion_control_oscbf.cpp)
set(TASK_SPACE_RRMC_SOURCES src/task_space_motion_control_rrmc.cpp)

# ---- MMC QP backend (OSQP wrapper) ----
add_library(osqp_solver STATIC
  src/osqp_solver.cpp
)
target_link_libraries(osqp_solver
  PUBLIC
    ${OSQP_TARGET}
)
target_include_directories(osqp_solver
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# Add executables
add_executable(robot_joint_dynamics ${ROBOT_JOINT_DYNAMICS_SOURCES})
add_executable(task_space_motion_control ${TASK_SPACE_MOTION_CONTROL_SOURCES})
add_executable(motion_reference_generator ${MOTION_REFERENCE_GENERATOR_SOURCES})
add_executable(task_space_motion_control_oscbf ${TASK_SPACE_OSCBF_SOURCES})
add_executable(task_space_motion_control_rrmc ${TASK_SPACE_RRMC_SOURCES})

# Link libraries
ament_target_dependencies(robot_joint_dynamics ${dependencies})
ament_target_dependencies(task_space_motion_control ${dependencies})
ament_target_dependencies(motion_reference_generator ${dependencies})
ament_target_dependencies(task_space_motion_control_oscbf ${dependencies})
ament_target_dependencies(task_space_motion_control_rrmc ${dependencies})

target_link_libraries(task_space_motion_control osqp_solver)
target_link_libraries(task_space_motion_control_oscbf osqp_solver)

# Install targets
install(TARGETS
  osqp_solver
  robot_joint_dynamics
  task_space_motion_control
  motion_reference_generator
  task_space_motion_control_oscbf
  task_space_motion_control_rrmc

  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY
  include/
  DESTINATION include/
)

install(
  DIRECTORY launch screw_lists rviz config
  DESTINATION share/${PROJECT_NAME}
)

# Linting and testing
if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
