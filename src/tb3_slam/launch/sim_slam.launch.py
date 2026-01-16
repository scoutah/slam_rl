from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
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

    return LaunchDescription([
        turtlebot3_gazebo_launch,
        turtlebot3_cartographer_launch_delayed
    ])
