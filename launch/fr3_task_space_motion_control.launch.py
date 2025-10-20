import os
import yaml
import xacro
from typing import List

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription, LaunchContext
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration

from launch_ros.actions import Node

""" 
Load yaml files and params
"""
yaml_dir = os.path.join(
    get_package_share_directory("motion_control"),
)
fr3_yaml = yaml_dir + "/screw_lists/fr3_body_screws.yaml"

initial_cfg_path = os.path.join(
    get_package_share_directory("motion_control"),
    "config", "initial_configuration.yaml"
)

motion_params = {
    "pos_x_cmd": 0.69, # 0.69
    "pos_y_cmd": 0.000, # 0.0
    "pos_z_cmd": 0.3, # 0.667, 0.3
    "quat_w_cmd": 0.172,
    "quat_x_cmd": 0.825,
    "quat_y_cmd": 0.342,
    "quat_z_cmd": 0.416,
    
    "offset_rad": [0.0],
    "amplitude_rad": [0.35],
    "frequency_hz": [0.2], # 0.05
    "phase_rad": [0.0],
}
    
def _load_yaml_dict(path: str) -> dict:
    with open(path, "r") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise RuntimeError(f"YAML at {path} is not a dict at top level.")
    return data

def _load_screw_list_params(yaml_path_abs: str) -> dict:
    params_raw = _load_yaml_dict(yaml_path_abs)
    robot_name = params_raw.get("robot_name", "arm")
    jl = params_raw.get("joint_limits", {})
    names = params_raw.get("joint_names", [])
    jl_lower  = []
    jl_upper  = []
    jl_vel    = []
    jl_effort = []
    for name in names:
        d = jl.get(name, {})
        jl_lower.append( float(d.get("lower",  0.0)) )
        jl_upper.append( float(d.get("upper",  0.0)) )
        jl_vel.append(   float(d.get("velocity", 0.0)) )
        jl_effort.append(float(d.get("effort",  0.0)) )

    screw_list_params = {
        "robot_name":           robot_name,
        "base_link":            params_raw.get("base_link", ""),
        "ee_link":              params_raw.get("ee_link", ""),
        "screw_representation": params_raw.get("screw_representation", "body"),
        "joint_names":          names,
        "num_joints":           params_raw.get("num_joints", 0),
        "screw_list":           params_raw.get("screw_list", {}),
        "joint_limits_lower":   jl_lower,
        "joint_limits_upper":   jl_upper,
        "joint_limits_velocity": jl_vel,
        "joint_limits_effort":  jl_effort,
        "M_position":           params_raw.get("M_position", []),
        "M_quaternion_wxyz":    params_raw.get("M_quaternion_wxyz", []),
    }
    return screw_list_params

def robot_state_publisher_spawner(context: LaunchContext, arm_id, load_gripper, ee_id):
    arm_id_str = context.perform_substitution(arm_id)
    load_gripper_str = context.perform_substitution(load_gripper)
    ee_id_str = context.perform_substitution(ee_id)
    franka_xacro_filepath = os.path.join(
        get_package_share_directory("franka_description"),
        "robots",
        arm_id_str,
        arm_id_str + ".urdf.xacro",
    )
    robot_description = xacro.process_file(
        franka_xacro_filepath, mappings={"hand": load_gripper_str, "ee_id": ee_id_str}
    ).toprettyxml(indent="  ")

    return [
        Node(
            package="robot_state_publisher",
            executable="robot_state_publisher",
            name="robot_state_publisher",
            output="screen",
            parameters=[{"robot_description": robot_description}],
        )
    ]


def rviz_spawner(context: LaunchContext):
    rviz_config_file = "fr3_motion_control.rviz"
    rviz_config_path = os.path.join(
        get_package_share_directory("motion_control"),
        "rviz", rviz_config_file
    )
    return [Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        arguments=["--display-config", rviz_config_path],
        output="screen",
    )]

def motion_reference_generator_spawner(context: LaunchContext) -> List[Node]:
    mrg  = Node(
        package="motion_control",
        executable="motion_reference_generator",
        name="motion_reference_generator",
        output="screen",
        parameters=[_load_screw_list_params(fr3_yaml),
                    {"gripper_joint_name": "fr3_finger_joint1"},
                    initial_cfg_path,
                    {"fs": 200.0},
                    motion_params],
        remappings=[("/pose_command", "/fr3/pose_command"),
                    ("/joint_command", "/fr3/joint_command")],
    )
    
    return [
            mrg, 
            ]

def motion_control_spawner(context: LaunchContext) -> List[Node]:
    # Task Space Motion Control
    tsmc = Node(
        package="motion_control",
        executable="task_space_motion_control",
        name="task_space_motion_control",
        output="screen",
        parameters=[_load_screw_list_params(fr3_yaml),
                    initial_cfg_path,
                    {"fs": 200.0}],
        remappings=[("/pose_command", "/fr3/pose_command"),
                    ("/joint_velocity_command", "/fr3/joint_velocity_command"),],
    )
    
    tsmc_oscbf = Node(
        package="motion_control",
        executable="task_space_motion_control_oscbf",
        name="task_space_motion_control_oscbf",
        output="screen",
        parameters=[_load_screw_list_params(fr3_yaml),
                    initial_cfg_path,
                    {"fs": 200.0}],
        remappings=[("/pose_command", "/fr3/pose_command"),
                    ("/joint_velocity_command", "/fr3/joint_velocity_command"),],
    )
    
    tsmc_rrmc = Node(
        package="motion_control",
        executable="task_space_motion_control_rrmc",
        name="task_space_motion_control_rrmc",
        output="screen",
        parameters=[_load_screw_list_params(fr3_yaml),
                    initial_cfg_path,
                    {"fs": 200.0}],
        remappings=[("/pose_command", "/fr3/pose_command"),
                    ("/joint_velocity_command", "/fr3/joint_velocity_command"),],
    )

    return [
            # tsmc, 
            tsmc_oscbf,
            # tsmc_rrmc,
            ]

def robot_joint_dynamics_spawner(context: LaunchContext) -> List[Node]:
    rjd = Node(
        package="motion_control",
        executable="robot_joint_dynamics",
        name="robot_joint_dynamics",
        output="screen",
        parameters=[_load_screw_list_params(fr3_yaml),
                    initial_cfg_path,
                    {"fs": 500.0},],
        remappings=[("/joint_velocity_command", "/fr3/joint_velocity_command")],
    )
    
    return [
            rjd, 
            ]

def generate_launch_description():
    # Args
    arm_id_arg = DeclareLaunchArgument(
        "arm_id",
        default_value="fr3",
        description="Franka arm type. Supported: fer, fr3, fp3, fr3v2",
    )
    load_gripper_arg = DeclareLaunchArgument(
        "load_gripper",
        default_value="true",
        description="Use end-effector if true; robot is loaded without EEF otherwise.",
    )
    ee_id_arg = DeclareLaunchArgument(
        "ee_id",
        default_value="franka_hand",
        description="End-effector ID. Supported: none, franka_hand, cobot_pump",
    )

    # LaunchConfigurations
    arm_id = LaunchConfiguration("arm_id")
    load_gripper = LaunchConfiguration("load_gripper")
    ee_id = LaunchConfiguration("ee_id")

    # Runtime-resolved spawners
    robot_state_publisher_loader = OpaqueFunction(
        function=robot_state_publisher_spawner,
        args=[arm_id, load_gripper, ee_id]
    )
    rviz_loader = OpaqueFunction(
        function=rviz_spawner,
        args=[]
    )
    motion_reference_generator_loader = OpaqueFunction(
        function=motion_reference_generator_spawner,
        args=[]
    )
    motion_control_loader = OpaqueFunction(
        function=motion_control_spawner,
        args=[]
    )
    robot_joint_dynamics_loader = OpaqueFunction(
        function=robot_joint_dynamics_spawner,
        args=[]
    )

    return LaunchDescription([
        arm_id_arg,
        load_gripper_arg,
        ee_id_arg,
        robot_state_publisher_loader,
        rviz_loader,
        motion_reference_generator_loader,
        motion_control_loader,
        robot_joint_dynamics_loader,
    ])
