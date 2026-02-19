import os

from ament_index_python import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, TimerAction, AppendEnvironmentVariable
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution, LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

from tb3_slam.utils import generate_random_world


def generate_launch_description():
    """
    This function generates a ROS 2 launch description that sets up a Gazebo simulation environment with a randomly generated world, spawns a TurtleBot3 robot, and launches the necessary nodes for SLAM and navigation (Cartographer and Nav2). It also includes a cmd_vel relay node to smooth the velocity commands sent to the robot.
    """


    # Configurations

    use_sim_time = LaunchConfiguration('use_sim_time', default='true')    
    world_file = os.path.join(get_package_share_directory('tb3_slam'), 'worlds', 'random_world.sdf')
    if not generate_random_world(world_file):
        exit(-1)
    
    # use gazebo world for testing only
    # world_file = os.path.join(get_package_share_directory('turtlebot3_gazebo'), 'worlds', 'turtlebot3_world.world')


    # Gazebo Bringup
    
    launch_file_dir = os.path.join(get_package_share_directory('turtlebot3_gazebo'), 'launch')
    ros_gz_sim = get_package_share_directory('ros_gz_sim')

    # Start Gazebo server with the specified world, but without the client (GUI)
    gzserver_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(ros_gz_sim, 'launch', 'gz_sim.launch.py')
        ),
        launch_arguments={'gz_args': ['-r -s -v2 ', world_file], 'on_exit_shutdown': 'true'}.items()
    )

    # gzclient_cmd = IncludeLaunchDescription(
    #     PythonLaunchDescriptionSource(
    #         os.path.join(ros_gz_sim, 'launch', 'gz_sim.launch.py')
    #     ),
    #     launch_arguments={'gz_args': '-g -v2 ', 'use_sim_time': use_sim_time}.items()
    # )

    robot_state_publisher_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(launch_file_dir, 'robot_state_publisher.launch.py')
        ),
        launch_arguments={'use_sim_time': use_sim_time}.items()
    )

    robot_state_publisher_delayed = TimerAction(
        period=2.0,
        actions=[robot_state_publisher_cmd]
    )

    spawn_turtlebot_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(launch_file_dir, 'spawn_turtlebot3.launch.py')
        ),
        launch_arguments={
            'x_pose': '2.0',
            'y_pose': '0.0',
            'use_sim_time': use_sim_time
        }.items()
    )

    spawn_turtlebot_cmd_delayed = TimerAction(
        period=2.0,
        actions=[spawn_turtlebot_cmd]
    )

    set_env_vars_resources = AppendEnvironmentVariable(
            'GZ_SIM_RESOURCE_PATH',
            os.path.join(
                get_package_share_directory('turtlebot3_gazebo'),
                'models'))


    # Cartographer and Nav2 Bringup

    turtlebot3_cartographer_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('turtlebot3_cartographer'),
                'launch',
                'cartographer.launch.py'
            ])
        ]),
        launch_arguments={'use_sim_time': use_sim_time}.items()
    )

    turtlebot3_cartographer_launch_delayed = TimerAction(
        period=5.0,
        actions=[turtlebot3_cartographer_launch]
    )

    nav2_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('nav2_bringup'),
                'launch',
                'navigation_launch.py'
            ])
        ]),
        launch_arguments={
            'use_sim_time': use_sim_time,
            'autostart': 'true'
        }.items()
    )

    nav2_launch_delayed = TimerAction(
        period=10.0,
        actions=[nav2_launch]
    )

    cmd_vel_relay = Node(
        package='tb3_slam',
        executable='cmd_vel_relay',
        name='cmd_vel_relay',
        parameters=[
            {
                'use_sim_time': use_sim_time,
                'input_topic': '/cmd_vel_smoothed',
                'output_topic': '/cmd_vel',
                'frame_id': 'base_link'
            }
        ]
    )

    return LaunchDescription([
        gzserver_cmd,
        # gzclient_cmd,
        spawn_turtlebot_cmd_delayed,
        robot_state_publisher_delayed,
        set_env_vars_resources,
        turtlebot3_cartographer_launch_delayed,
        nav2_launch_delayed,
        cmd_vel_relay
    ])
