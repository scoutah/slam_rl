from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    turtlebot3_gazebo_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('turtlebot3_gazebo'),
                'launch',
                'turtlebot3_world.launch.py'
            ])
        ])
    )

    turtlebot3_cartographer_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('turtlebot3_cartographer'),
                'launch',
                'cartographer.launch.py'
            ])
        ]),
        launch_arguments={'use_sim_time': 'true'}.items()
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
            'use_sim_time': 'true',
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
                'use_sim_time': True,
                'input_topic': '/cmd_vel_smoothed',
                'output_topic': '/cmd_vel',
                'frame_id': 'base_link'
            }
        ]
    )

    return LaunchDescription([
        turtlebot3_gazebo_launch,
        turtlebot3_cartographer_launch_delayed,
        nav2_launch_delayed,
        cmd_vel_relay
    ])
