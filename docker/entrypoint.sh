#!/usr/bin/env bash
# Entrypoint for the SLAM-RL container.
#
# Sources the ROS 2 distro and the colcon workspace overlay, exports the
# TurtleBot3 model and the RL data directory, then execs whatever command was
# passed (defaults to an interactive shell via the Dockerfile CMD).
set -e

WORKSPACE="${WORKSPACE:-/root/projects/slam_rl}"

# ROS 2 base + our built workspace overlay.
source /opt/ros/jazzy/setup.bash
if [ -f "${WORKSPACE}/install/setup.bash" ]; then
    source "${WORKSPACE}/install/setup.bash"
fi

# Required by TurtleBot3 launch/description files.
export TURTLEBOT3_MODEL="${TURTLEBOT3_MODEL:-burger}"

# Persist the RL model/replay buffer to a stable, mountable location.
# rl_agent.py reads SLAM_RL_DATA_DIR first (see _resolve_persist_dir); defaulting
# it here points persistence at <workspace>/rl_data, which is a mounted volume.
export SLAM_RL_DATA_DIR="${SLAM_RL_DATA_DIR:-${WORKSPACE}/rl_data}"
mkdir -p "${SLAM_RL_DATA_DIR}"

exec "$@"
