import os
from pathlib import Path
import shlex
import sys
import tempfile

from ament_index_python import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    AppendEnvironmentVariable,
    ExecuteProcess,
    GroupAction,
    IncludeLaunchDescription,
    OpaqueFunction,
    SetEnvironmentVariable,
    TimerAction,
)
from launch.conditions import UnlessCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node, PushRosNamespace
from tb3_slam.multi_robot import (
    bridge_config,
    frame,
    map_merge_params,
    MERGED_MAP_FRAME,
    namespace_cartographer_lua,
    namespace_model_sdf,
    namespace_nav2_params,
    ROBOT_NAMESPACES,
    topic,
)
from tb3_slam.utils import (
    check_world_size,
    DEFAULT_WORLD_SIZE,
    generate_random_world,
    load_spawns,
    WORLD_META_FILE,
    world_meta_path,
)
import yaml

# Every robot-local node sees its own TF tree on /<ns>/tf (the Nav2
# multi-robot convention); frames are additionally prefixed (<ns>/base_link)
# so the trees can later be joined without collisions.
TF_REMAPS = [('/tf', 'tf'), ('/tf_static', 'tf_static')]

CARTOGRAPHER_LUA = 'turtlebot3_lds_2d.lua'


def prepare_world(world_size: float) -> tuple[str, list[tuple[float, float]]]:
    """Return (world_file, robot_spawns) for this launch.

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

    # The generator writes both robot spawns (side by side at the room edge,
    # scale with world size) into the world metadata; load_spawns falls back
    # to the default geometry if a pre-made world has no fresh metadata.
    return world_file, load_spawns(WORLD_META_FILE)


def write_robot_configs(ns: str, out_dir: Path) -> dict[str, str]:
    """Write the namespaced model SDF, bridge, Cartographer and Nav2 configs.

    Returns the paths keyed by 'model', 'bridge', 'cartographer_dir', 'nav2'.
    """
    tb3_gazebo = Path(get_package_share_directory('turtlebot3_gazebo'))
    robot_dir = out_dir / ns
    robot_dir.mkdir(parents=True, exist_ok=True)

    model = robot_dir / 'model.sdf'
    model.write_text(namespace_model_sdf(
        (tb3_gazebo / 'models' / 'turtlebot3_burger' / 'model.sdf').read_text(), ns,
    ))

    bridge = robot_dir / 'bridge.yaml'
    bridge.write_text(yaml.safe_dump(bridge_config(ns), sort_keys=False))

    # Only the robot-specific lua lives here; Cartographer resolves its
    # includes (map_builder.lua, ...) from its own installed config dir.
    carto_src = (Path(get_package_share_directory('turtlebot3_cartographer'))
                 / 'config' / CARTOGRAPHER_LUA)
    (robot_dir / CARTOGRAPHER_LUA).write_text(
        namespace_cartographer_lua(carto_src.read_text(), ns)
    )

    nav2_src = Path(get_package_share_directory('nav2_bringup')) / 'params' / 'nav2_params.yaml'
    nav2 = robot_dir / 'nav2_params.yaml'
    nav2.write_text(yaml.safe_dump(
        namespace_nav2_params(yaml.safe_load(nav2_src.read_text()), ns), sort_keys=False,
    ))

    return {
        'model': str(model),
        'bridge': str(bridge),
        'cartographer_dir': str(robot_dir),
        'nav2': str(nav2),
    }


def robot_stack(ns: str, spawn: tuple[float, float], configs: dict[str, str],
                use_sim_time) -> list:
    """Timer-sequenced simulation stack of one robot inside namespace ns.

    spawn + bridge → 2 s robot_state_publisher → 5 s Cartographer
    → 6 s startup spin → 10 s Nav2. cmd_vel_relay starts immediately.
    """
    spawn_x, spawn_y = spawn

    spawn_robot = Node(
        package='ros_gz_sim',
        executable='create',
        arguments=[
            '-name', f'burger_{ns}',
            '-file', configs['model'],
            '-x', str(spawn_x),
            '-y', str(spawn_y),
            '-z', '0.01',
        ],
        output='screen',
    )

    bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        name='gz_bridge',
        namespace=ns,
        parameters=[{'config_file': configs['bridge'], 'use_sim_time': use_sim_time}],
        output='log',
    )

    urdf_path = os.path.join(
        get_package_share_directory('turtlebot3_gazebo'), 'urdf', 'turtlebot3_burger.urdf'
    )
    with open(urdf_path) as f:
        robot_desc = f.read()

    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        namespace=ns,
        parameters=[{
            'use_sim_time': use_sim_time,
            'robot_description': robot_desc,
            'frame_prefix': f'{ns}/',
        }],
        remappings=TF_REMAPS,
        output='log',
    )

    cartographer = Node(
        package='cartographer_ros',
        executable='cartographer_node',
        name='cartographer_node',
        namespace=ns,
        parameters=[{'use_sim_time': use_sim_time}],
        arguments=['-configuration_directory', configs['cartographer_dir'],
                   '-configuration_basename', CARTOGRAPHER_LUA],
        remappings=TF_REMAPS,
        output='log',
    )
    occupancy_grid = Node(
        package='cartographer_ros',
        executable='cartographer_occupancy_grid_node',
        name='cartographer_occupancy_grid_node',
        namespace=ns,
        parameters=[{'use_sim_time': use_sim_time}],
        arguments=['-resolution', '0.05', '-publish_period_sec', '1.0'],
        output='log',
    )

    # Publishes TwistStamped for 4 s so Cartographer can map free cells
    # before the exploration nodes start looking for frontiers.
    startup_spin = ExecuteProcess(
        cmd=[
            'ros2', 'topic', 'pub',
            '--rate', '10',
            '--times', '40',
            topic(ns, 'cmd_vel'),
            'geometry_msgs/msg/TwistStamped',
            f"{{header: {{frame_id: {frame(ns, 'base_link')}}}, twist: {{angular: {{z: 1.5}}}}}}",
        ],
        output='log',
    )

    # navigation_launch.py does not push the namespace itself (bringup_launch
    # does), so wrap it; it remaps /tf -> tf and roots the params at ns.
    nav2 = GroupAction([
        PushRosNamespace(ns),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(
                get_package_share_directory('nav2_bringup'), 'launch', 'navigation_launch.py',
            )),
            launch_arguments={
                'namespace': ns,
                'params_file': configs['nav2'],
                'use_sim_time': use_sim_time,
                'autostart': 'true',
            }.items(),
        ),
    ])

    # Nav2 /<ns>/cmd_vel_smoothed (Twist) → /<ns>/cmd_vel (TwistStamped, gz bridge).
    cmd_vel_relay = Node(
        package='tb3_slam',
        executable='cmd_vel_relay',
        name='cmd_vel_relay',
        namespace=ns,
        output='log',
        parameters=[{
            'use_sim_time': use_sim_time,
            'input_topic': topic(ns, 'cmd_vel_smoothed'),
            'output_topic': topic(ns, 'cmd_vel'),
            'frame_id': frame(ns, 'base_link'),
        }],
    )

    return [
        spawn_robot,
        bridge,
        cmd_vel_relay,
        TimerAction(period=2.0, actions=[robot_state_publisher]),
        TimerAction(period=5.0, actions=[cartographer, occupancy_grid]),
        TimerAction(period=6.0, actions=[startup_spin]),
        TimerAction(period=10.0, actions=[nav2]),
    ]


def map_merge_stack(spawns: list[tuple[float, float]], use_sim_time) -> list:
    """Merge the robots' maps into one global ``/map`` (issue #9).

    ``multirobot_map_merge`` (built from source, see tools/install_map_merge.sh)
    runs with known initial poses = the spawns and fuses the grids with a
    per-cell max, which keeps the 0-25 / 26-74 / 75-100 / -1 semantics that
    frontier_utils relies on. It publishes no TF, so static ``map -> <ns>/map``
    transforms join the robots' trees to the merged frame.
    """
    merge = Node(
        package='multirobot_map_merge',
        executable='map_merge',
        name='map_merge',
        output='log',
        parameters=[{
            'use_sim_time': use_sim_time,
            'known_init_poses': True,
            'robot_map_topic': 'map',
            'robot_map_updates_topic': 'map_updates',
            'merged_map_topic': 'map',
            'world_frame': MERGED_MAP_FRAME,
            'merging_rate': 4.0,
            # Default is one discovery pass per 20 s; the maps appear ~5 s
            # after launch and the merged map should follow within seconds.
            'discovery_rate': 0.5,
            **map_merge_params(spawns),
        }],
    )
    static_tfs = [
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name=f'static_tf_{ns}_map',
            arguments=[
                '--x', str(x), '--y', str(y), '--z', '0',
                '--yaw', '0', '--pitch', '0', '--roll', '0',
                '--frame-id', MERGED_MAP_FRAME,
                '--child-frame-id', frame(ns, 'map'),
            ],
            parameters=[{'use_sim_time': use_sim_time}],
            output='log',
        )
        for ns, (x, y) in zip(ROBOT_NAMESPACES, spawns)
    ]
    # Cartographer starts at 5 s; the first /<ns>/map follows its first
    # publish period, so merging needs no earlier start.
    return [*static_tfs, TimerAction(period=7.0, actions=[merge])]


def explorer_node(ns: str, package: str, executable: str, name: str,
                  use_sim_time, coverage_threshold, episode_file) -> Node:
    """Single-robot explorer running inside namespace ns on that robot's map.

    Interim until the allocator (#11) assigns goals on the merged map (#9);
    the explorers still use their own robot's map, the episode end does not.
    """
    return Node(
        package=package,
        executable=executable,
        name=name,
        namespace=ns,
        ros_arguments=['--log-level', 'info'],
        parameters=[{
            'use_sim_time': use_sim_time,
            'coverage_threshold': coverage_threshold,
            'episode_file': episode_file,
            'map_frame': frame(ns, 'map'),
            'base_frame': frame(ns, 'base_link'),
        }],
        # The nodes use absolute names; resolve them inside the namespace.
        remappings=TF_REMAPS + [
            ('/map', 'map'),
            ('/cmd_vel', 'cmd_vel'),
            ('/frontiers_viz', 'frontiers_viz'),
        ],
    )


def generate_launch_description():
    use_sim_time = LaunchConfiguration('use_sim_time', default='true')
    headless = LaunchConfiguration('headless', default='false')
    exploration_mode = LaunchConfiguration('exploration_mode', default='classic')
    episode_file = LaunchConfiguration('episode_file', default='/tmp/slam_episode_done')
    coverage_threshold = LaunchConfiguration('coverage_threshold', default='0.99')
    max_episode_duration = LaunchConfiguration('max_episode_duration', default='600.0')
    world_size = LaunchConfiguration('world_size', default=str(DEFAULT_WORLD_SIZE))

    ros_gz_sim = get_package_share_directory('ros_gz_sim')

    # ── World + Gazebo server + both robot stacks ───────────────────────
    # The world is generated at launch time (not description time) so the
    # world_size launch argument can be resolved first; the robot stacks
    # need the spawns from that world's metadata.

    def launch_world(context):
        world_file, spawns = prepare_world(
            check_world_size(float(world_size.perform(context)))
        )
        gzserver_cmd = IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(ros_gz_sim, 'launch', 'gz_sim.launch.py')
            ),
            launch_arguments={
                'gz_args': ['-r -s -v2 ', shlex.quote(world_file)],
                'on_exit_shutdown': 'true',
            }.items(),
        )
        # One global sim clock for all nodes (the robot bridges skip it).
        clock_bridge = Node(
            package='ros_gz_bridge',
            executable='parameter_bridge',
            name='clock_bridge',
            arguments=['/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock'],
            output='log',
        )
        config_dir = Path(tempfile.mkdtemp(prefix='slam_rl_robots_'))
        robots = []
        for ns, spawn in zip(ROBOT_NAMESPACES, spawns):
            robots += robot_stack(ns, spawn, write_robot_configs(ns, config_dir), use_sim_time)
        return [gzserver_cmd, clock_bridge, *robots, *map_merge_stack(spawns, use_sim_time)]

    # ── Gazebo GUI ──────────────────────────────────────────────────────

    gzclient_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(ros_gz_sim, 'launch', 'gz_sim.launch.py')
        ),
        launch_arguments={'gz_args': '-g -v2 ', 'use_sim_time': use_sim_time}.items(),
        condition=UnlessCondition(headless),
    )

    set_env_vars_resources = AppendEnvironmentVariable(
        'GZ_SIM_RESOURCE_PATH',
        os.path.join(get_package_share_directory('turtlebot3_gazebo'), 'models'),
    )

    # ── Episode manager ──────────────────────────────────────────────────
    # Runs on the merged global /map published by map_merge_stack().

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
    # Start with a delay so SLAM + Nav2 are ready first.

    def launch_explorers(context):
        mode = exploration_mode.perform(context)
        if mode == 'classic':
            return [
                explorer_node(ns, 'tb3_exploration', 'frontier_explorer', 'frontier_explorer',
                              use_sim_time, coverage_threshold, episode_file)
                for ns in ROBOT_NAMESPACES
            ]
        if mode == 'rl':
            # The DQN persists to one shared rl_data/ dir, so only robot 1 runs
            # it until the centralised allocator (#13) replaces it.
            return [explorer_node(ROBOT_NAMESPACES[0], 'tb3_reinforced_exploration',
                                  'reinforced_exploration', 'rl_agent',
                                  use_sim_time, coverage_threshold, episode_file)]
        if mode == 'none':
            # Bare stack: goals come from outside (manual tests, allocator).
            return []
        raise ValueError(f"exploration_mode must be 'classic', 'rl' or 'none' (got {mode!r})")

    # The RL agent needs PyTorch which lives in the project-local .venv.
    # We inject the venv's site-packages into PYTHONPATH so the system Python
    # (used by ros2 launch) can import torch without a global install.
    # colcon sets COLCON_PREFIX_PATH to <workspace>/install; go one level up.
    _ws_root = os.path.dirname(os.environ.get('COLCON_PREFIX_PATH', ''))
    _py_ver = f'python{sys.version_info.major}.{sys.version_info.minor}'
    _venv_site = os.path.join(_ws_root, '.venv', 'lib', _py_ver, 'site-packages')

    inject_venv = AppendEnvironmentVariable('PYTHONPATH', _venv_site)

    exploration_delayed = TimerAction(
        period=20.0,
        actions=[OpaqueFunction(function=launch_explorers)],
    )

    return LaunchDescription([
        # Suppress verbose logs from Cartographer, Nav2, robot_state_publisher etc.
        # Only WARN and above from external nodes will appear on the terminal.
        SetEnvironmentVariable('RCUTILS_LOGGING_SEVERITY_THRESHOLD', 'WARN'),
        set_env_vars_resources,
        inject_venv,
        OpaqueFunction(function=launch_world),
        gzclient_cmd,
        episode_manager,
        exploration_delayed,
    ])
