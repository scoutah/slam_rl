"""Tests for per-robot namespacing of the TurtleBot3 configs."""
import copy

import pytest

from tb3_slam.multi_robot import (
    bridge_config,
    frame,
    MERGED_MAP_FRAME,
    namespace_cartographer_lua,
    namespace_model_sdf,
    namespace_nav2_params,
    ROBOT_NAMESPACES,
    topic,
)

SDF_SNIPPET = """<?xml version="1.0" ?>
<sdf version="1.8">
  <model name="turtlebot3_burger">
    <sensor name="tb3_imu" type="imu">
      <always_on>true</always_on>
      <update_rate>200</update_rate>
      <topic>imu</topic>
    </sensor>
    <sensor name="hls_lfcd_lds" type="gpu_lidar">
      <update_rate>5</update_rate>
      <topic>scan</topic>
      <gz_frame_id>base_scan</gz_frame_id>
    </sensor>
    <plugin filename="gz-sim-diff-drive-system" name="gz::sim::systems::DiffDrive">
      <topic>cmd_vel</topic>
      <odom_topic>odom</odom_topic>
      <frame_id>odom</frame_id>
      <child_frame_id>base_footprint</child_frame_id>
      <tf_topic>/tf</tf_topic>
    </plugin>
    <plugin filename="gz-sim-joint-state-publisher-system"
            name="gz::sim::systems::JointStatePublisher">
      <topic>joint_states</topic>
    </plugin>
  </model>
</sdf>
"""

LUA_SNIPPET = """include "map_builder.lua"

options = {
  map_builder = MAP_BUILDER,
  map_frame = "map",
  tracking_frame = "imu_link",
  published_frame = "odom",
  odom_frame = "odom",
  provide_odom_frame = false,
}
"""

NAV2_SNIPPET = {
    'amcl': {'ros__parameters': {'base_frame_id': 'base_footprint', 'odom_frame_id': 'odom'}},
    'local_costmap': {
        'local_costmap': {
            'ros__parameters': {
                'global_frame': 'odom',
                'robot_base_frame': 'base_link',
                'footprint_topic': '/local_costmap/published_footprint',
                'plugins': ['voxel_layer'],
                'voxel_layer': {
                    'scan': {'topic': '/scan', 'sensor_frame': 'base_scan'},
                },
            },
        },
    },
    'global_costmap': {
        'global_costmap': {
            'ros__parameters': {
                'global_frame': 'map',
                'robot_base_frame': 'base_link',
                'obstacle_layer': {'scan': {'topic': '/scan'}},
            },
        },
    },
    'bt_navigator': {
        'ros__parameters': {'odom_topic': '/odom', 'global_frame': 'map'},
    },
    'collision_monitor': {
        'ros__parameters': {
            'base_frame_id': 'base_footprint',
            'cmd_vel_in_topic': 'cmd_vel_smoothed',
            'observation_sources': ['scan'],
            'scan': {'topic': 'scan'},
        },
    },
    'docking': {'docks': [{'frame': 'base_link', 'pose': [0.0, 0.0]}]},
}


def test_frame_joins_namespace_and_name():
    assert frame('r1', 'base_link') == 'r1/base_link'


def test_topic_adds_leading_slash_and_namespace():
    assert topic('r1', 'scan') == '/r1/scan'


def test_topic_accepts_absolute_name():
    assert topic('r1', '/scan') == '/r1/scan'


def test_robot_namespaces_match_num_robots():
    assert ROBOT_NAMESPACES == ('r1', 'r2')


def test_sdf_topics_are_namespaced():
    out = namespace_model_sdf(SDF_SNIPPET, 'r1')

    assert '<tf_topic>/r1/tf</tf_topic>' in out
    assert '<topic>/r1/scan</topic>' in out
    assert '<topic>/r1/imu</topic>' in out
    assert '<topic>/r1/cmd_vel</topic>' in out
    assert '<odom_topic>/r1/odom</odom_topic>' in out
    assert '<topic>/r1/joint_states</topic>' in out


def test_sdf_frames_are_namespaced():
    out = namespace_model_sdf(SDF_SNIPPET, 'r1')

    assert '<child_frame_id>r1/base_footprint</child_frame_id>' in out
    assert '<frame_id>r1/odom</frame_id>' in out
    assert '<gz_frame_id>r1/base_scan</gz_frame_id>' in out


def test_sdf_other_elements_untouched():
    out = namespace_model_sdf(SDF_SNIPPET, 'r1')

    assert '<update_rate>200</update_rate>' in out
    assert '<always_on>true</always_on>' in out
    assert '<sensor name="tb3_imu" type="imu">' in out
    assert '<plugin filename="gz-sim-diff-drive-system"' in out


def test_sdf_only_changes_rewritten_elements():
    out = namespace_model_sdf(SDF_SNIPPET, 'r1')
    before = SDF_SNIPPET.splitlines()
    after = out.splitlines()

    assert len(before) == len(after)
    changed = [b for b, a in zip(before, after) if a != b]
    assert len(changed) == 9  # 9 topic/frame elements in SDF_SNIPPET
    assert all('topic>' in line or 'frame_id>' in line for line in changed)


def test_sdf_tolerates_surrounding_whitespace():
    sdf = '<tf_topic>  /tf \n</tf_topic>'
    out = namespace_model_sdf(sdf, 'r2')

    assert out == '<tf_topic>  /r2/tf \n</tf_topic>'


def test_sdf_two_namespaces_are_disjoint():
    out1 = namespace_model_sdf(SDF_SNIPPET, 'r1')
    out2 = namespace_model_sdf(SDF_SNIPPET, 'r2')

    assert '/r1/' not in out2
    assert 'r1/' not in out2
    assert '/r2/' not in out1
    assert 'r2/' not in out1


def test_sdf_without_tf_topic_raises():
    with pytest.raises(ValueError):
        namespace_model_sdf('<model><topic>cmd_vel</topic></model>', 'r1')


def test_bridge_has_no_clock_and_all_topics_namespaced():
    entries = bridge_config('r1')

    assert all(e['ros_topic_name'].startswith('/r1/') for e in entries)
    assert all(e['gz_topic_name'] == e['ros_topic_name'] for e in entries)
    assert not any(e['ros_topic_name'].endswith('clock') for e in entries)
    assert {e['ros_topic_name'] for e in entries} == {
        '/r1/joint_states', '/r1/odom', '/r1/tf', '/r1/cmd_vel', '/r1/imu', '/r1/scan',
    }


def test_bridge_entries_have_expected_keys_and_types():
    entries = {e['ros_topic_name']: e for e in bridge_config('r1')}

    assert set(entries['/r1/scan']) == {
        'ros_topic_name', 'gz_topic_name', 'ros_type_name', 'gz_type_name', 'direction',
    }
    assert entries['/r1/scan']['gz_type_name'] == 'gz.msgs.LaserScan'
    assert entries['/r1/cmd_vel']['direction'] == 'ROS_TO_GZ'
    assert entries['/r1/cmd_vel']['ros_type_name'] == 'geometry_msgs/msg/TwistStamped'


def test_bridge_two_robots_are_disjoint():
    names1 = {e['ros_topic_name'] for e in bridge_config('r1')}
    names2 = {e['ros_topic_name'] for e in bridge_config('r2')}

    assert names1.isdisjoint(names2)


def test_cartographer_frames_are_prefixed():
    out = namespace_cartographer_lua(LUA_SNIPPET, 'r1')

    assert '  map_frame = "r1/map",' in out
    assert '  tracking_frame = "r1/imu_link",' in out
    assert '  published_frame = "r1/odom",' in out
    assert '  odom_frame = "r1/odom",' in out


def test_cartographer_leaves_other_options_unchanged():
    out = namespace_cartographer_lua(LUA_SNIPPET, 'r1')

    assert 'provide_odom_frame = false,' in out
    assert 'map_builder = MAP_BUILDER,' in out
    assert 'include "map_builder.lua"' in out


def test_cartographer_missing_option_raises():
    lua = LUA_SNIPPET.replace('  odom_frame = "odom",\n', '')

    with pytest.raises(ValueError, match='odom_frame'):
        namespace_cartographer_lua(lua, 'r1')


def test_nav2_frames_are_prefixed():
    out = namespace_nav2_params(NAV2_SNIPPET, 'r1')

    assert out['local_costmap']['local_costmap']['ros__parameters']['global_frame'] == 'r1/odom'
    assert out['global_costmap']['global_costmap']['ros__parameters']['global_frame'] == 'r1/map'
    assert (out['local_costmap']['local_costmap']['ros__parameters']['robot_base_frame']
            == 'r1/base_link')
    assert out['amcl']['ros__parameters']['base_frame_id'] == 'r1/base_footprint'
    assert out['amcl']['ros__parameters']['odom_frame_id'] == 'r1/odom'
    assert out['docking']['docks'][0]['frame'] == 'r1/base_link'


def test_nav2_absolute_topics_are_namespaced():
    out = namespace_nav2_params(NAV2_SNIPPET, 'r1')
    local = out['local_costmap']['local_costmap']['ros__parameters']

    assert local['voxel_layer']['scan']['topic'] == '/r1/scan'
    assert local['footprint_topic'] == '/r1/local_costmap/published_footprint'
    assert out['bt_navigator']['ros__parameters']['odom_topic'] == '/r1/odom'
    assert (out['global_costmap']['global_costmap']['ros__parameters']['obstacle_layer']
            ['scan']['topic'] == '/r1/scan')


def test_nav2_relative_topics_unchanged():
    out = namespace_nav2_params(NAV2_SNIPPET, 'r1')
    collision = out['collision_monitor']['ros__parameters']

    assert collision['cmd_vel_in_topic'] == 'cmd_vel_smoothed'
    assert collision['scan']['topic'] == 'scan'


def test_nav2_lists_are_traversed_and_unrelated_values_kept():
    out = namespace_nav2_params(NAV2_SNIPPET, 'r1')
    docks = out['docking']['docks']

    assert docks[0]['pose'] == [0.0, 0.0]
    assert (out['local_costmap']['local_costmap']['ros__parameters']['plugins']
            == ['voxel_layer'])


def test_nav2_input_not_mutated():
    original = copy.deepcopy(NAV2_SNIPPET)

    namespace_nav2_params(NAV2_SNIPPET, 'r1')

    assert NAV2_SNIPPET == original


def test_nav2_idempotent():
    once = namespace_nav2_params(NAV2_SNIPPET, 'r1')
    twice = namespace_nav2_params(once, 'r1')

    assert twice == once


def test_merged_map_frame_is_not_a_robot_frame():
    assert all(frame(ns, 'map') != MERGED_MAP_FRAME for ns in ROBOT_NAMESPACES)
