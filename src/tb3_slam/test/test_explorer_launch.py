"""Check standalone method selection and headless TF availability."""
import importlib.util
from pathlib import Path

from launch import LaunchContext
from launch.utilities import normalize_to_list_of_substitutions, perform_substitutions
from launch_ros.actions import Node
import pytest


def _launch_module():
    path = Path(__file__).parents[1] / 'launch' / 'sim_slam.launch.py'
    spec = importlib.util.spec_from_file_location('explorer_launch_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('mode, package, executable', [
    ('classic', 'tb3_exploration', 'classic_explorer'),
    ('rl', 'tb3_reinforced_exploration', 'rl_explorer'),
])
def test_mode_selects_its_standalone_explorer(mode, package, executable):
    node = _launch_module().explorer_node(mode, True, 0.99, '/tmp/episode')
    context = LaunchContext()
    assert perform_substitutions(
        context, normalize_to_list_of_substitutions(node.node_package)) == package
    assert perform_substitutions(
        context, normalize_to_list_of_substitutions(node.node_executable)) == executable


def test_headless_exploration_receives_both_robot_tf_trees(monkeypatch):
    module = _launch_module()
    lookup = module.get_package_share_directory
    monkeypatch.setattr(module, 'get_package_share_directory', lambda package:
                        str(Path(__file__).parents[1])
                        if package == 'tb3_slam' else lookup(package))
    description = module.generate_launch_description()
    context = LaunchContext()
    context.launch_configurations.update(headless='true', rviz='false')
    relay = next(node for node in description.entities if isinstance(node, Node)
                 and perform_substitutions(context, normalize_to_list_of_substitutions(
                     node.node_executable)) == 'tf_relay')
    assert relay.condition is None or relay.condition.evaluate(context)


def test_spawn_transform_is_visible_to_each_robots_nav2(monkeypatch):
    module = _launch_module()
    monkeypatch.setattr(module, 'Node', lambda **kwargs: kwargs)
    nodes = module.map_merge_stack([(0.0, -3.0), (1.0, -3.0)], 8.0, True)
    transforms = [node for node in nodes if node['executable'] == 'static_transform_publisher']
    assert len(transforms) == 2
    for ns, node in zip(('r1', 'r2'), transforms):
        assert ('/tf_static', f'/{ns}/tf_static') in node.get('remappings', [])
