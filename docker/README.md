# Docker – headless SLAM-RL

Containerised, **headless** TurtleBot3 exploration/training. No GUI (no Gazebo
client, no RViz) – this is meant for development on macOS and for training runs
on AWS. Visual debugging still needs a native Linux desktop or a VM.

The same image runs on amd64 (Linux / AWS EC2) and arm64 (Apple-Silicon Mac).

## Prerequisites

- Docker (Docker Desktop on macOS, Docker Engine on Linux).
- A clone of this repository.

## Quick start

```bash
# Build the image (first time, or after changing the Dockerfile)
docker compose build

# Run 20 classic-exploration episodes, headless
docker compose run --rm train --mode classic --episodes 20 --headless

# Run the RL agent
docker compose run --rm train --mode rl --episodes 50 --headless

# Same, but Gazebo up to 4x faster than real time (CPU-bound; 0 = unlimited)
docker compose run --rm train --mode rl --episodes 50 --headless --sim-speed 4

# Interactive shell (for colcon build, tests, debugging)
docker compose run --rm dev
```

`results/`, `worlds/` and `rl_data/` are bind-mounted from your checkout, so
CSVs, generated worlds and the trained model appear directly in your local repo.

## Editing code

`src/`, `training_runner.py` and `tools/` are mounted into the container. Edit
on the host, then rebuild the ROS workspace inside the container:

```bash
docker compose run --rm dev
# inside the container:
colcon build --symlink-install
source install/setup.bash
python3 training_runner.py --mode rl --episodes 5 --headless
```

Only changing the Dockerfile (new apt/pip deps) requires `docker compose build`.

## Apple Silicon note

The image builds and runs natively as arm64. Headless Gazebo physics runs on
CPU, so it works but is slower than an amd64 box – fine for development and
correctness checks, not for large training runs (use AWS for those).

## AWS (headless training)

1. Build and push to ECR (CPU instances are the right choice – the DQN is tiny,
   the bottleneck is Gazebo physics on CPU, so no GPU is needed):

   ```bash
   aws ecr get-login-password --region <region> \
     | docker login --username AWS --password-stdin <acct>.dkr.ecr.<region>.amazonaws.com
   docker build -t slam-rl:latest .
   docker tag slam-rl:latest <acct>.dkr.ecr.<region>.amazonaws.com/slam-rl:latest
   docker push <acct>.dkr.ecr.<region>.amazonaws.com/slam-rl:latest
   ```

2. On an EC2 instance (e.g. a compute-optimised `c7i`/`c6i`) with Docker:

   ```bash
   docker run --init --rm \
     -v "$PWD/results:/root/projects/slam_rl/results" \
     -v "$PWD/rl_data:/root/projects/slam_rl/rl_data" \
     <acct>.dkr.ecr.<region>.amazonaws.com/slam-rl:latest \
     python3 training_runner.py --mode rl --episodes 100 --headless
   ```

   `--init` matters: the runner spawns and kills a process tree each episode;
   the init reaper prevents zombie buildup over long runs.

> For cost estimates use the [AWS Pricing Calculator](https://calculator.aws/);
> actual cost depends on instance type, region and runtime.

## Why `init: true` / `--init`?

`training_runner.py` launches `ros2 launch` (Gazebo, Nav2, Cartographer,
bridges) in a new session and tears it down with `pkill` between episodes.
Without a PID-1 init process to reap exited children, zombies accumulate over
many episodes. Compose sets `init: true`; with plain `docker run` pass `--init`.
