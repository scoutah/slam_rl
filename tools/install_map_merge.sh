#!/usr/bin/env bash
# Build multirobot_map_merge (ROS 2 port of m-explore) from source.
#
# The package has no apt binary. It is built into its own overlay workspace
# (default ~/m-explore-ws, override with M_EXPLORE_WS) at a pinned commit, so
# the project workspace stays untouched. Used by the Dockerfile and by native
# Linux setups; afterwards `source $M_EXPLORE_WS/install/setup.bash` BEFORE
# the project's install/setup.bash.
#
# To update the pin: pick a commit of robo-friends/m-explore-ros2, verify that
# a two-robot episode still merges, then change M_EXPLORE_REF below.
set -euo pipefail

M_EXPLORE_REF="${M_EXPLORE_REF:-326cf8a0b487c34246bb8f3326afbcd69576dc60}"
M_EXPLORE_WS="${M_EXPLORE_WS:-$HOME/m-explore-ws}"
ROS_SETUP="${ROS_SETUP:-/opt/ros/jazzy/setup.bash}"

mkdir -p "${M_EXPLORE_WS}/src"
if [ ! -d "${M_EXPLORE_WS}/src/m-explore-ros2/.git" ]; then
    git clone https://github.com/robo-friends/m-explore-ros2.git \
        "${M_EXPLORE_WS}/src/m-explore-ros2"
fi
git -C "${M_EXPLORE_WS}/src/m-explore-ros2" fetch --quiet origin
git -C "${M_EXPLORE_WS}/src/m-explore-ros2" checkout --quiet "${M_EXPLORE_REF}"

# shellcheck disable=SC1090
source "${ROS_SETUP}"
cd "${M_EXPLORE_WS}"
rosdep install --from-paths src/m-explore-ros2/map_merge --ignore-src -y -r
colcon build --packages-select multirobot_map_merge
echo "Built multirobot_map_merge at ${M_EXPLORE_REF} in ${M_EXPLORE_WS}"
