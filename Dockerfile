# ─────────────────────────────────────────────────────────────────────────────
# TurtleBot3 SLAM-RL – headless simulation / training image
#
# Base: ROS 2 Jazzy (Ubuntu 24.04, Python 3.12) – multi-arch (amd64 + arm64),
# so the same Dockerfile builds on an amd64 Linux host / AWS EC2 and on an
# Apple-Silicon (arm64) Mac.
#
# The image is HEADLESS only: Gazebo runs its server, no GUI client, no RViz.
# This matches `training_runner.py --headless` and avoids any X11 dependency.
# ─────────────────────────────────────────────────────────────────────────────
FROM ros:jazzy-ros-base

# Non-interactive apt + unbuffered Python logs (so training output streams out)
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    TURTLEBOT3_MODEL=burger

# ─────────────────────────────────────────────────────────────────────────────
# System + ROS dependencies
#
# The three ament_python packages only declare rclpy / *_msgs in package.xml;
# the simulation stack (Gazebo, Nav2, Cartographer, TurtleBot3) is NOT in
# package.xml and must be installed explicitly – mirroring the README's apt
# install list. python3-venv is needed to build the torch venv below.
# ─────────────────────────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3-pip \
        python3-venv \
        python3-colcon-common-extensions \
        ros-jazzy-turtlebot3 \
        ros-jazzy-turtlebot3-simulations \
        ros-jazzy-cartographer-ros \
        ros-jazzy-nav2-bringup \
        ros-jazzy-ros-gz \
        ros-jazzy-tf-transformations \
    && rm -rf /var/lib/apt/lists/*

# ─────────────────────────────────────────────────────────────────────────────
# Project-local virtualenv with PyTorch (CPU build).
#
# The launch file injects `<workspace>/.venv/lib/python3.12/site-packages` into
# PYTHONPATH so the system Python used by `ros2 launch` can import torch.
# We therefore create the venv at exactly that path. --system-site-packages
# lets it see the ROS Python packages too. CPU-only torch: the DQN is tiny and
# the real bottleneck is the Gazebo physics (CPU), so no CUDA is needed.
# ─────────────────────────────────────────────────────────────────────────────
ENV WORKSPACE=/root/projects/slam_rl
WORKDIR ${WORKSPACE}

RUN python3 -m venv --system-site-packages "${WORKSPACE}/.venv" \
    && "${WORKSPACE}/.venv/bin/pip" install --no-cache-dir --upgrade pip \
    && "${WORKSPACE}/.venv/bin/pip" install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cpu \
        torch

# ─────────────────────────────────────────────────────────────────────────────
# Copy sources and build the colcon workspace.
#
# Only src/ is needed to build; results/worlds/rl_data are runtime volumes.
# .dockerignore keeps the host .venv/build/install out of the build context.
# ─────────────────────────────────────────────────────────────────────────────
COPY src/ ${WORKSPACE}/src/
COPY training_runner.py ${WORKSPACE}/
COPY tools/ ${WORKSPACE}/tools/

RUN . /opt/ros/jazzy/setup.sh \
    && colcon build --symlink-install

# ─────────────────────────────────────────────────────────────────────────────
# Entrypoint sources ROS + the workspace overlay and sets env, then execs the
# given command (default: an interactive bash shell).
# ─────────────────────────────────────────────────────────────────────────────
COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["bash"]
