# SLAM RL Installationsanleitung

## ROS2 Jazzy Jalisco Installation: 
- https://docs.ros.org/en/jazzy/Installation/Ubuntu-Install-Debs.html

## Gazebo Sim Installation: 
- https://gazebosim.org/docs/jetty/install_ubuntu/

## TurtleBot3 Installation: 
```bash
sudo apt update
sudo apt install \
  ros-jazzy-turtlebot3 \
  ros-jazzy-turtlebot3-simulations \
  ros-jazzy-slam-toolbox
```

## Starting The Simulation

Cloning the repository
```bash
cd your-projects-folder

# clone via ssh
git clone git@github.com:scoutah/slam_rl.git
# or clone via https
git clone https://github.com/scoutah/slam_rl.git

cd slam_rl
```

Starting the simulation
```bash
# build files creation
colcon build # ONLY EVER USE THAT COMMAND IN 'slam_rl/', NOT IN slam_rl/src or other subdirectories
source install/setup.bash

# launch simulation
ros2 launch tb3_slam sim_slam.launch.py
```

Controlling the simulation via keyboard (in a seperat terminal)
```bash
# control simulation via keyboard
ros2 run turtlebot3_teleop teleop_keyboard
```
