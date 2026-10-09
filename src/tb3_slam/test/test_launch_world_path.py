"""Check world paths at the Gazebo shell-command boundary."""
import importlib.util
import json
from pathlib import Path
import shlex
import subprocess

from launch import LaunchContext
from launch.actions import OpaqueFunction
from launch.utilities import normalize_to_list_of_substitutions, perform_substitutions
import pytest


@pytest.mark.parametrize('world_file', [
    '/tmp/random_world.sdf',
    '/mnt/c/Users/lboehlen/Studienprojekt 1/slam_rl/worlds/random_world.sdf',
    "/tmp/robot's worlds/room $1.sdf",
])
def test_gazebo_receives_world_path_as_one_literal_argument(monkeypatch, world_file):
    launch_path = Path(__file__).parents[1] / 'launch' / 'sim_slam.launch.py'
    spec = importlib.util.spec_from_file_location('sim_slam_launch_path_test', launch_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, 'prepare_world', lambda size: (world_file, 0.0, 0.0))
    description = module.generate_launch_description()
    context = LaunchContext()
    world_action = next(action for action in description.entities
                        if isinstance(action, OpaqueFunction))
    server = world_action.execute(context)[0]
    arguments = dict(server.launch_arguments)
    gz_args = perform_substitutions(
        context, normalize_to_list_of_substitutions(arguments['gz_args']),
    )
    assert shlex.split(gz_args) == ['-r', '-s', '-v2', world_file]
    # Match ros_gz_sim's shell=True boundary, including shell expansions.
    probe = shlex.quote('import json, sys; print(json.dumps(sys.argv[1:]))')
    result = subprocess.run(
        f'/usr/bin/python3 -c {probe} {gz_args}',
        shell=True, check=True, capture_output=True, text=True,
    )
    assert json.loads(result.stdout) == ['-r', '-s', '-v2', world_file]
