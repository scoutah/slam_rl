# ~/ros2_ws/run_turtlebot3_slam.sh
#!/bin/bash
# Workspace sourcen
source ~/ros2_ws/install/setup.bash
# TurtleBot3 Modell setzen
export TURTLEBOT3_MODEL=burger
# Gazebo starten
ros2 launch turtlebot3_gazebo turtlebot3_world.launch.py &
sleep 5
# SLAM starten
ros2 launch turtlebot3_cartographer cartographer.launch.py use_sim_time:=true &
sleep 5
# RViz starten
rviz2 &
# Teleop (im Vordergrund)
ros2 run turtlebot3_teleop teleop_keyboard
