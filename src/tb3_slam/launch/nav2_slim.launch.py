"""Slimmed Nav2 bringup: only the nodes this project's explorer actually uses.

Upstream ``nav2_bringup/navigation_launch.py`` lifecycle-manages 10 nodes; each
configure→activate step (and the lifecycle_manager bond to it) adds to the ~30 s
r1 bringup. The frontier explorer only needs ``navigate_to_pose`` (bt_navigator),
``compute_path_to_pose`` (planner_server), a controller and recovery behaviours to
drive, and velocity_smoother — whose output ``cmd_vel_smoothed`` the repo's
``cmd_vel_relay`` consumes directly (so collision_monitor's passthrough is bypassed
and it, like route/waypoint/docking and the path smoother, is dropped). Fewer nodes
through the lifecycle = faster, more reliable activation.

No Nav2 lifecycle_manager is launched (``autostart`` is accepted but unused):
``nav2_gate`` drives these nodes through
configure/activate itself, with per-call timeouts and state re-checks. Nav2's
manager waits forever on each ``change_state`` response, and Fast DDS drops
such a response when the caller's reply reader is not matched yet under load
("failed to send response ... client will not receive response") — that hung
r2's bringup even with the robots serialised.

Drop-in for navigation_launch.py (same launch args: namespace, use_sim_time,
params_file, autostart, use_respawn, log_level); composition is not supported here.
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction, SetEnvironmentVariable
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node, SetParameter
from launch_ros.descriptions import ParameterFile
from nav2_common.launch import RewrittenYaml


# The nodes below must match tb3_slam.multi_robot.NAV2_LIFECYCLE_NODES, the
# list (and configure/activate order) nav2_gate brings up.
def generate_launch_description():
    namespace = LaunchConfiguration('namespace')
    use_sim_time = LaunchConfiguration('use_sim_time')
    autostart = LaunchConfiguration('autostart')
    params_file = LaunchConfiguration('params_file')
    use_respawn = LaunchConfiguration('use_respawn')
    log_level = LaunchConfiguration('log_level')

    # Same /tf remapping as upstream (the node namespace is pushed by the caller).
    remappings = [('/tf', 'tf'), ('/tf_static', 'tf_static')]

    configured_params = ParameterFile(
        RewrittenYaml(
            source_file=params_file,
            root_key=namespace,
            param_rewrites={'autostart': autostart},
            convert_types=True,
        ),
        allow_substs=True,
    )

    stdout_linebuf_envvar = SetEnvironmentVariable(
        'RCUTILS_LOGGING_BUFFERED_STREAM', '1'
    )

    def nav_node(package, executable, name, extra_remaps=None):
        return Node(
            package=package,
            executable=executable,
            name=name,
            output='screen',
            respawn=use_respawn,
            respawn_delay=2.0,
            parameters=[configured_params],
            arguments=['--ros-args', '--log-level', log_level],
            remappings=remappings + (extra_remaps or []),
        )

    load_nodes = GroupAction(actions=[
        SetParameter('use_sim_time', use_sim_time),
        nav_node('nav2_controller', 'controller_server', 'controller_server',
                 [('cmd_vel', 'cmd_vel_nav')]),
        nav_node('nav2_planner', 'planner_server', 'planner_server'),
        nav_node('nav2_behaviors', 'behavior_server', 'behavior_server',
                 [('cmd_vel', 'cmd_vel_nav')]),
        nav_node('nav2_bt_navigator', 'bt_navigator', 'bt_navigator'),
        nav_node('nav2_velocity_smoother', 'velocity_smoother', 'velocity_smoother',
                 [('cmd_vel', 'cmd_vel_nav')]),
    ])

    ld = LaunchDescription()
    ld.add_action(stdout_linebuf_envvar)
    ld.add_action(DeclareLaunchArgument('namespace', default_value=''))
    ld.add_action(DeclareLaunchArgument('use_sim_time', default_value='false'))
    ld.add_action(DeclareLaunchArgument(
        'params_file',
        default_value=os.path.join(
            get_package_share_directory('nav2_bringup'), 'params', 'nav2_params.yaml'),
    ))
    ld.add_action(DeclareLaunchArgument('autostart', default_value='true'))
    ld.add_action(DeclareLaunchArgument('use_respawn', default_value='False'))
    ld.add_action(DeclareLaunchArgument('log_level', default_value='info'))
    ld.add_action(load_nodes)
    return ld
