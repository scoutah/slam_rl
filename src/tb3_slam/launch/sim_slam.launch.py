"""Launch the full two-robot SLAM/exploration stack (see README code map for the order)."""
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
    RegisterEventHandler,
    SetEnvironmentVariable,
    TimerAction,
)
from launch.conditions import IfCondition, UnlessCondition
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node, PushRosNamespace
from tb3_slam.multi_robot import (
    bridge_config,
    frame,
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
    load_world_size,
    WORLD_META_FILE,
    world_meta_path,
)
import yaml

# Every robot-local node sees its own TF tree on /<ns>/tf (the Nav2
# multi-robot convention); frames are additionally prefixed (<ns>/base_link)
# so the trees can later be joined without collisions.
TF_REMAPS = [('/tf', 'tf'), ('/tf_static', 'tf_static')]

CARTOGRAPHER_LUA = 'turtlebot3_lds_2d.lua'

# Each robot runs spawn→bridge→RSP→Cartographer→startup-spin in parallel; Nav2 is
# then brought up per robot via nav2_slim.launch.py (only the 5 nodes the explorer
# uses — see that file). The two Nav2 bringups run near-parallel but with a small
# per-robot configure offset (NAV2_BASE_DELAY + i*NAV2_CONFIGURE_OFFSET) so their
# lifecycle change_state phases never overlap — that overlap was the sole cause of
# r2's intermittent change_state stall. One nav2_gate process per robot blocks on
# that robot's navigate_to_pose; the explorer launches (OnProcessExit countdown)
# the instant the last gate exits, i.e. when both stacks are active — no fixed
# explorer delay.
#
# Pre-Nav2 nodes use short fixed timers (no lifecycle race): RSP at 2 s,
# Cartographer at 5 s, startup spin at 6 s; r1 Nav2 at 6 s, r2 at 10 s.
GATE_TIMEOUT_S = 120.0  # nav2_gate gives up (exit 1) if Nav2 never comes up


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
        namespace_nav2_params(yaml.safe_load(nav2_src.read_text()), ns, MERGED_MAP_FRAME),
        sort_keys=False,
    ))

    return {
        'model': str(model),
        'bridge': str(bridge),
        'cartographer_dir': str(robot_dir),
        'nav2': str(nav2),
    }


def robot_stack(ns: str, spawn: tuple[float, float], configs: dict[str, str],
                use_sim_time) -> tuple[list, GroupAction]:
    """Build one robot's simulation stack inside namespace ``ns``.

    Returns ``(pre_nav2_actions, nav2)``: the pre-Nav2 actions start immediately
    (spawn + bridge + cmd_vel_relay now; RSP @2 s, Cartographer @5 s, startup
    spin @6 s via short fixed timers — these have no lifecycle race). The ``nav2``
    GroupAction is returned separately so ``launch_world`` can serialise the two
    robots' Nav2 bringups behind readiness gates instead of starting it here.
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

    # Publishes TwistStamped for 12.5 s (125 msgs @10 Hz, 0.5 rad/s ≈ one turn)
    # so Cartographer can map free cells before the explorer looks for frontiers.
    startup_spin = ExecuteProcess(
        cmd=[
            'ros2', 'topic', 'pub',
            '--rate', '10',
            '--times', '125',
            topic(ns, 'cmd_vel'),
            'geometry_msgs/msg/TwistStamped',
            f"{{header: {{frame_id: {frame(ns, 'base_link')}}}, twist: {{angular: {{z: 0.5}}}}}}",
        ],
        output='log',
    )

    # nav2_slim.launch.py (our trimmed bringup) does not push the namespace
    # itself, so wrap it; it remaps /tf -> tf and roots the params at ns.
    nav2 = GroupAction([
        PushRosNamespace(ns),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(
                get_package_share_directory('tb3_slam'), 'launch', 'nav2_slim.launch.py',
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

    return (
        [
            spawn_robot,
            bridge,
            cmd_vel_relay,
            TimerAction(period=2.0, actions=[robot_state_publisher]),
            TimerAction(period=5.0, actions=[cartographer, occupancy_grid]),
            TimerAction(period=6.0, actions=[startup_spin]),
        ],
        nav2,
    )


def nav2_gate_process(ns: str) -> ExecuteProcess:
    """Short-lived process that blocks until ``<ns>``'s Nav2 is active.

    Exits 0 the moment ``/<ns>/navigate_to_pose`` is available, exit 1 on
    timeout. Used to chain the robots' Nav2 bringups (OnProcessExit) so they
    serialise on a real readiness signal instead of a fixed wall-clock offset.
    """
    return ExecuteProcess(
        cmd=['ros2', 'run', 'tb3_slam', 'nav2_gate',
             '--ns', ns, '--timeout', str(GATE_TIMEOUT_S)],
        name=f'nav2_gate_{ns}',
        output='screen',
    )


def map_merge_stack(spawns: list[tuple[float, float]], world_size: float,
                    use_sim_time) -> list:
    """Merge the robots' maps into one global ``/map`` (issue #9).

    ``map_merger`` places every ``/<ns>/map`` by its origin plus the spawn pose
    on a fixed-extent grid and fuses with a per-cell max (keeps the 0-25 /
    26-74 / 75-100 / -1 semantics frontier_utils relies on). Static
    ``map -> <ns>/map`` transforms join the robots' TF trees to the merged frame.
    """
    merge = Node(
        package='tb3_slam',
        executable='map_merger',
        name='map_merger',
        output='log',
        parameters=[{
            'use_sim_time': use_sim_time,
            'spawns': [float(v) for spawn in spawns for v in spawn],
            'world_size': float(world_size),
        }],
    )
    static_tfs = [
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name=f'static_tf_{ns}_map{suffix}',
            arguments=[
                '--x', str(x), '--y', str(y), '--z', '0',
    # The same offsets also go onto each robot's own /<ns>/tf_static so Nav2's
    # global costmap (frame ``map``) can resolve map -> <ns>/map.
                '--yaw', '0', '--pitch', '0', '--roll', '0',
                '--frame-id', MERGED_MAP_FRAME,
                '--child-frame-id', frame(ns, 'map'),
            ],
            parameters=[{'use_sim_time': use_sim_time}],
            output='log',
        )
        for ns, (x, y) in zip(ROBOT_NAMESPACES, spawns)
    ]
    return [*static_tfs, merge]


            remappings=remap,
def explorer_node(mode, use_sim_time, coverage_threshold, episode_file) -> Node:
    """Standalone explorer node for the selected method.

        for suffix, remap in (
            ('', []),
            ('_robot_tree', [('/tf_static', topic(ns, 'tf_static'))]),
        )
    ``classic`` runs the cost-utility explorer in ``tb3_exploration``; ``rl``
    runs the learned explorer in ``tb3_reinforced_exploration``. Each is its own
    node reading the merged global ``/map`` and dispatching goals to the
    per-robot ``/<ns>/navigate_to_pose`` actions; the methods share only the
    infrastructure in the base node, not a strategy.
    """
    package, executable = {
        'classic': ('tb3_exploration', 'classic_explorer'),
        'rl': ('tb3_reinforced_exploration', 'rl_explorer'),
    }[mode]
    return Node(
        package=package,
        executable=executable,
        name=f'{mode}_explorer',
        ros_arguments=['--log-level', 'info'],
        parameters=[{
            'use_sim_time': use_sim_time,
            'coverage_threshold': coverage_threshold,
            'episode_file': episode_file,
        }],
    )


def generate_launch_description():
    use_sim_time = LaunchConfiguration('use_sim_time', default='true')
    headless = LaunchConfiguration('headless', default='false')
    # RViz opens with the Gazebo GUI; rviz:=false switches it off, headless never opens it.
    rviz = LaunchConfiguration('rviz', default='true')
    exploration_mode = LaunchConfiguration('exploration_mode', default='classic')
    episode_file = LaunchConfiguration('episode_file', default='/tmp/slam_episode_done')
    coverage_threshold = LaunchConfiguration('coverage_threshold', default='0.97')
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

        # Each robot's pre-Nav2 phase starts in parallel; their Nav2 bringups are
        # serialised behind readiness gates so they never configure concurrently.
        pre_nav2_actions: list = []
        nav2_by_ns: dict[str, GroupAction] = {}
        for ns, spawn in zip(ROBOT_NAMESPACES, spawns):
            pre, nav2 = robot_stack(
                ns, spawn, write_robot_configs(ns, config_dir), use_sim_time,
            )
            pre_nav2_actions += pre
            nav2_by_ns[ns] = nav2

        # Nav2 bringups run (almost) in parallel for speed, but their lifecycle
        # *configure* phases are decoupled by a small per-robot offset so they
        # never issue change_state concurrently — that overlap was the only
        # cause of r2's intermittent change_state stall, and it is just the
        # first few seconds of each bringup, not the whole thing. Waiting for
        # r1 to be fully active (the earlier serial design) needlessly delayed
        # r2 by ~25 s; a small offset removes the race at near-zero cost.
        #
        # The explorer still starts on a real readiness signal: one nav2_gate
        # per robot blocks on that robot's navigate_to_pose, and the explorer
        # launches once *both* gates have exited (whichever finishes last).
        NAV2_BASE_DELAY = 6.0         # r1 Nav2 right after Cartographer (5 s) sets up map->odom TF
        NAV2_CONFIGURE_OFFSET = 4.0   # stagger per robot so configure phases don't overlap

        chain: list = []
        gates: list[ExecuteProcess] = []
        for i, ns in enumerate(ROBOT_NAMESPACES):
            chain.append(TimerAction(
                period=NAV2_BASE_DELAY + i * NAV2_CONFIGURE_OFFSET,
                actions=[nav2_by_ns[ns]],
            ))
            gates.append(nav2_gate_process(ns))

        # Start all gates immediately (they just block on their action server),
        # and launch the explorer when the last one still running exits. A tiny
        # countdown closure fires launch_explorers on the final gate exit.
        remaining = {'n': len(gates)}

        def _on_gate_exit(*_args, **_kwargs):
            remaining['n'] -= 1
            if remaining['n'] == 0:
                return [OpaqueFunction(function=launch_explorers)]
            return []

        chain += gates
        for gate in gates:
            chain.append(RegisterEventHandler(OnProcessExit(
                target_action=gate,
                on_exit=_on_gate_exit,
            )))

        return [
            gzserver_cmd, clock_bridge, *pre_nav2_actions, *chain,
            *map_merge_stack(spawns, load_world_size(WORLD_META_FILE), use_sim_time),
        ]

    # ── Gazebo GUI ──────────────────────────────────────────────────────

    gzclient_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(ros_gz_sim, 'launch', 'gz_sim.launch.py')
        ),
        launch_arguments={'gz_args': '-g -v2 ', 'use_sim_time': use_sim_time}.items(),
        condition=UnlessCondition(headless),
    )

    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', os.path.join(
            get_package_share_directory('tb3_slam'), 'rviz', 'sim_slam.rviz')],
        parameters=[{'use_sim_time': use_sim_time}],
        output='log',
        condition=IfCondition(PythonExpression([
            "'", rviz, "'.lower() == 'true' and '", headless, "'.lower() != 'true'",
        ])),
    )

    # RViz displays like RobotModel can only subscribe to the global /tf, but
    # every robot keeps its transforms on /<ns>/tf. This relay mirrors the
    # (prefixed, collision-free) per-robot trees onto /tf for RViz only; it is
    # gated on the same condition as RViz and never feeds a robot stack.
    tf_relay_node = Node(
        package='tb3_slam',
        executable='tf_relay',
        name='tf_relay',
        parameters=[{
            'use_sim_time': use_sim_time,
            'namespaces': list(ROBOT_NAMESPACES),
        }],
        output='log',
        condition=IfCondition(PythonExpression([
            "'", rviz, "'.lower() == 'true' and '", headless, "'.lower() != 'true'",
        ])),
    )

    set_env_vars_resources = AppendEnvironmentVariable(
        'GZ_SIM_RESOURCE_PATH',
        os.path.join(get_package_share_directory('turtlebot3_gazebo'), 'models'),
    )

    # ── Episode manager ──────────────────────────────────────────────────
    # ── Episode manager ──────────────────────────────────────────────────
    # Runs on the merged global /map published by map_merge_stack(). The
    # episode timeout is derived from the world size by the node itself
    # (issue #10) and is not configurable here.

    episode_manager = Node(
        package='tb3_slam',
        executable='episode_manager',
        name='episode_manager',
        ros_arguments=['--log-level', 'info'],
        parameters=[{
            'use_sim_time': use_sim_time,
            'episode_file': episode_file,
            'coverage_threshold': coverage_threshold,
        }],
    )

    # ── Exploration nodes (mutually exclusive) ───────────────────────────
    # Start with a delay so SLAM + Nav2 are ready first.

    def launch_explorers(context):
        mode = exploration_mode.perform(context)
        # `classic` and `rl` are the user-facing methods, each a standalone
        # explorer node on the merged map; they share only the base node's
        # infrastructure. `none` brings up the bare stack (goals from outside).
        if mode in ('classic', 'rl'):
            return [explorer_node(
                mode, use_sim_time, coverage_threshold, episode_file,
            )]
        if mode == 'none':
            return []
        raise ValueError(
            f"exploration_mode must be 'classic', 'rl' or 'none' (got {mode!r})"
        )

    # The RL agent needs PyTorch from the project-local .venv; inject its
    # site-packages into PYTHONPATH. Resolve the workspace root from
    # COLCON_PREFIX_PATH when the shell was sourced, else from this installed
    # launch file's location (<ws>/install/tb3_slam/share/tb3_slam/launch/).
    _py_ver = f'python{sys.version_info.major}.{sys.version_info.minor}'
    _prefix = os.environ.get('COLCON_PREFIX_PATH', '').split(os.pathsep)[0]
    if _prefix:
        _ws_root = os.path.dirname(_prefix)
    else:
        _ws_root = str(Path(__file__).resolve().parents[5])
    _venv_site = os.path.join(_ws_root, '.venv', 'lib', _py_ver, 'site-packages')
    if not os.path.isdir(_venv_site):
        print(
            f'[sim_slam.launch] WARNING: venv site-packages not found at '
            f'{_venv_site} — the RL explorer will fail to import torch '
            f'(see AGENTS.md).',
            file=sys.stderr,
        )

    inject_venv = AppendEnvironmentVariable('PYTHONPATH', _venv_site)

    # The explorer is no longer time-triggered: launch_world chains it onto the
    # last robot's Nav2 readiness gate (OnProcessExit), so it starts the instant
    # both robots' Nav2 stacks are active instead of after a fixed delay.

    return LaunchDescription([
        # Suppress verbose logs from Cartographer, Nav2, robot_state_publisher etc.
        # Only WARN and above from external nodes will appear on the terminal.
        SetEnvironmentVariable('RCUTILS_LOGGING_SEVERITY_THRESHOLD', 'WARN'),
        set_env_vars_resources,
        inject_venv,
        OpaqueFunction(function=launch_world),
        gzclient_cmd,
        rviz_node,
        tf_relay_node,
        episode_manager,
    ])
