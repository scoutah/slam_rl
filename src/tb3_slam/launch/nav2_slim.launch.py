"""Slimmed Nav2 bringup: only the nodes this project's explorer actually uses.

Upstream ``nav2_bringup/navigation_launch.py`` lifecycle-manages 10 nodes; each
configure→activate step (and the lifecycle_manager bond to it) adds to the ~30 s
r1 bringup. The frontier explorer only needs ``navigate_to_pose`` (bt_navigator),
``compute_path_to_pose`` (planner_server), a controller and recovery behaviours to
drive, and velocity_smoother — whose output ``cmd_vel_smoothed`` the repo's
``cmd_vel_relay`` consumes directly (so collision_monitor's passthrough is bypassed
and it, like route/waypoint/docking and the path smoother, is dropped). Fewer nodes
through the lifecycle = faster, more reliable activation.

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

# Only these go through the lifecycle manager (order = configure/activate order).
# Everything the explorer never calls (smoother_server, route_server,
# collision_monitor, waypoint_follower, docking_server) is left out on purpose.
LIFECYCLE_NODES = [
    'controller_server',
    'planner_server',
    'behavior_server',
    'bt_navigator',
    'velocity_smoother',
]


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
        Node(
            package='nav2_lifecycle_manager',
            executable='lifecycle_manager',
            name='lifecycle_manager_navigation',
            output='screen',
            arguments=['--ros-args', '--log-level', log_level],
            parameters=[{'autostart': autostart}, {'node_names': LIFECYCLE_NODES}],
        ),
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
