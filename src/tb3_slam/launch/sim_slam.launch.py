import os
import sys

from ament_index_python import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, TimerAction, AppendEnvironmentVariable, ExecuteProcess, OpaqueFunction, SetEnvironmentVariable
from launch.conditions import IfCondition, UnlessCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution, LaunchConfiguration, PythonExpression
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

from tb3_slam.utils import (
    DEFAULT_WORLD_SIZE,
    WALL_THICKNESS,
    check_world_size,
    generate_random_world,
    WORLD_META_FILE,
    spawn_pose,
    world_meta_path,
)


def prepare_world(world_size: float) -> tuple[str, float, float]:
    """Return (world_file, spawn_x, spawn_y) for this launch.

    Uses SLAM_WORLD_FILE if set (world_size is then ignored – a saved world
    has its size baked in), otherwise generates a random world of world_size.
    """
    world_file_val = os.environ.get('SLAM_WORLD_FILE', '')
    if world_file_val:
        world_file = world_file_val
        # Reusing a saved world: its metadata (coverage denominator + spawn)
        # lives in a sidecar next to the SDF, not in the shared /tmp file which
        # still describes the last *generated* world. Copy the sidecar into
        # WORLD_META_FILE so coverage_monitor and the spawn logic below read the
        # values for the world actually loaded.
        sidecar = world_meta_path(world_file)
        if sidecar.exists():
            WORLD_META_FILE.write_text(sidecar.read_text())
        else:
            print(
                f'[launch] WARNING: no metadata sidecar for {world_file} '
                f'({sidecar}); coverage denominator may be wrong.',
                file=sys.stderr,
            )
    else:
        world_file = os.path.join(
            get_package_share_directory('tb3_slam'), 'worlds', 'random_world.sdf'
        )
        success, _expected_cells = generate_random_world(world_file, world_size)
        if not success:
            exit(-1)

    # The generator writes the robot spawn (edge of the room, scales with world
    # size) into the world metadata. Fall back to the default world geometry if
    # a pre-made world is used without fresh metadata.
    try:
        import json
        meta = json.loads(WORLD_META_FILE.read_text())
        spawn_x = float(meta['spawn_x'])
        spawn_y = float(meta['spawn_y'])
    except (OSError, ValueError, KeyError):
        spawn_x, spawn_y = spawn_pose(DEFAULT_WORLD_SIZE, WALL_THICKNESS)
    return world_file, spawn_x, spawn_y


def generate_launch_description():
    use_sim_time = LaunchConfiguration('use_sim_time', default='true')
    headless = LaunchConfiguration('headless', default='false')
    exploration_mode = LaunchConfiguration('exploration_mode', default='classic')
    episode_file = LaunchConfiguration('episode_file', default='/tmp/slam_episode_done')
    coverage_threshold = LaunchConfiguration('coverage_threshold', default='0.99')
    max_episode_duration = LaunchConfiguration('max_episode_duration', default='600.0')
    world_size = LaunchConfiguration('world_size', default=str(DEFAULT_WORLD_SIZE))

    launch_file_dir = os.path.join(get_package_share_directory('turtlebot3_gazebo'), 'launch')
    ros_gz_sim = get_package_share_directory('ros_gz_sim')

    # ── World + Gazebo server + robot spawn ─────────────────────────────
    # The world is generated at launch time (not description time) so the
    # world_size launch argument can be resolved first.

    def launch_world(context):
        world_file, spawn_x, spawn_y = prepare_world(
            check_world_size(float(world_size.perform(context)))
        )
        gzserver_cmd = IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(ros_gz_sim, 'launch', 'gz_sim.launch.py')
            ),
            launch_arguments={
                'gz_args': ['-r -s -v2 ', world_file],
                'on_exit_shutdown': 'true',
            }.items(),
        )
        spawn_turtlebot_cmd = IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(launch_file_dir, 'spawn_turtlebot3.launch.py')
            ),
            launch_arguments={
                'x_pose': str(spawn_x),
                'y_pose': str(spawn_y),
                'use_sim_time': use_sim_time,
            }.items(),
        )
        return [gzserver_cmd, spawn_turtlebot_cmd]

    # ── Gazebo GUI ──────────────────────────────────────────────────────

    gzclient_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(ros_gz_sim, 'launch', 'gz_sim.launch.py')
        ),
        launch_arguments={'gz_args': '-g -v2 ', 'use_sim_time': use_sim_time}.items(),
        condition=UnlessCondition(headless),
    )

    robot_state_publisher_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(launch_file_dir, 'robot_state_publisher.launch.py')
        ),
        launch_arguments={'use_sim_time': use_sim_time}.items(),
    )

    robot_state_publisher_delayed = TimerAction(period=2.0, actions=[robot_state_publisher_cmd])

    set_env_vars_resources = AppendEnvironmentVariable(
        'GZ_SIM_RESOURCE_PATH',
        os.path.join(get_package_share_directory('turtlebot3_gazebo'), 'models'),
    )

    # ── SLAM & Navigation ────────────────────────────────────────────────

    turtlebot3_cartographer_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('turtlebot3_cartographer'),
                'launch',
                'cartographer.launch.py',
            ])
        ]),
        launch_arguments={'use_sim_time': use_sim_time}.items(),
    )

    turtlebot3_cartographer_launch_delayed = TimerAction(
        period=5.0, actions=[turtlebot3_cartographer_launch]
    )

    nav2_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('nav2_bringup'),
                'launch',
                'navigation_launch.py',
            ])
        ]),
        launch_arguments={
            'use_sim_time': use_sim_time,
            'autostart': 'true',
        }.items(),
    )

    nav2_launch_delayed = TimerAction(period=10.0, actions=[nav2_launch])

    # ── Startup spin (after Cartographer, before Nav2) ───────────────────
    # Publishes TwistStamped to /cmd_vel for 4 s so Cartographer can map
    # free cells before the exploration nodes start looking for frontiers.

    startup_spin = ExecuteProcess(
        cmd=[
            'ros2', 'topic', 'pub',
            '--rate', '10',
            '--times', '40',
            '/cmd_vel',
            'geometry_msgs/msg/TwistStamped',
            '{header: {frame_id: base_link}, twist: {angular: {z: 1.5}}}',
        ],
        output='log',
    )
    startup_spin_delayed = TimerAction(period=6.0, actions=[startup_spin])

    # ── Velocity relay ───────────────────────────────────────────────────

    cmd_vel_relay = Node(
        package='tb3_slam',
        executable='cmd_vel_relay',
        name='cmd_vel_relay',
        output='log',
        parameters=[{
            'use_sim_time': use_sim_time,
            'input_topic': '/cmd_vel_smoothed',
            'output_topic': '/cmd_vel',
            'frame_id': 'base_link',
        }],
    )

    # ── Episode manager ──────────────────────────────────────────────────

    episode_manager = Node(
        package='tb3_slam',
        executable='episode_manager',
        name='episode_manager',
        ros_arguments=['--log-level', 'info'],
        parameters=[{
            'use_sim_time': use_sim_time,
            'episode_file': episode_file,
            'coverage_threshold': coverage_threshold,
            'max_episode_duration': max_episode_duration,
        }],
    )

    # ── Exploration nodes (mutually exclusive) ───────────────────────────
    # Start both with a delay so SLAM + Nav2 are ready first.

    frontier_explorer = Node(
        package='tb3_exploration',
        executable='frontier_explorer',
        name='frontier_explorer',
        ros_arguments=['--log-level', 'info'],
        parameters=[{
            'use_sim_time': use_sim_time,
            'coverage_threshold': coverage_threshold,
            'episode_file': episode_file,
        }],
        condition=IfCondition(
            PythonExpression(["'", exploration_mode, "' == 'classic'"])
        ),
    )

    # The RL agent needs PyTorch which lives in the project-local .venv.
    # We inject the venv's site-packages into PYTHONPATH so the system Python
    # (used by ros2 launch) can import torch without a global install.
    # colcon sets COLCON_PREFIX_PATH to <workspace>/install; go one level up.
    _ws_root = os.path.dirname(os.environ.get('COLCON_PREFIX_PATH', ''))
    _py_ver = f'python{sys.version_info.major}.{sys.version_info.minor}'
    _venv_site = os.path.join(_ws_root, '.venv', 'lib', _py_ver, 'site-packages')

    inject_venv = AppendEnvironmentVariable('PYTHONPATH', _venv_site)

    rl_agent = Node(
        package='tb3_reinforced_exploration',
        executable='reinforced_exploration',
        name='rl_agent',
        ros_arguments=['--log-level', 'info'],
        parameters=[{
            'use_sim_time': use_sim_time,
            'coverage_threshold': coverage_threshold,
            'episode_file': episode_file,
        }],
        condition=IfCondition(
            PythonExpression(["'", exploration_mode, "' == 'rl'"])
        ),
    )

    exploration_delayed = TimerAction(
        period=20.0,
        actions=[frontier_explorer, rl_agent],
    )

    return LaunchDescription([
        # Suppress verbose logs from Cartographer, Nav2, robot_state_publisher etc.
        # Only WARN and above from external nodes will appear on the terminal.
        SetEnvironmentVariable('RCUTILS_LOGGING_SEVERITY_THRESHOLD', 'WARN'),
        set_env_vars_resources,
        inject_venv,
        OpaqueFunction(function=launch_world),
        gzclient_cmd,
        robot_state_publisher_delayed,
        turtlebot3_cartographer_launch_delayed,
        startup_spin_delayed,
        nav2_launch_delayed,
        cmd_vel_relay,
        episode_manager,
        exploration_delayed,
    ])
