"""Per-robot namespacing of the single-robot TurtleBot3 configs.

The two-robot simulation runs every robot under its own namespace (``r1``,
``r2``). The upstream TurtleBot3 Gazebo model, bridge, Cartographer and Nav2
configs use global names; the helpers here rewrite them so each robot gets its
own topics and frames. Pure Python (stdlib only) so they can be unit tested
without ROS.
"""
from collections.abc import Callable
import copy
import re

from tb3_slam.utils import NUM_ROBOTS

ROBOT_NAMESPACES: tuple[str, ...] = tuple(f'r{i + 1}' for i in range(NUM_ROBOTS))

# Frame (and topic name) of the merged global map; each robot's own map frame
# ``<ns>/map`` is a child of it, offset by the robot's spawn pose (yaw 0).
MERGED_MAP_FRAME: str = 'map'

# Gazebo model SDF elements holding topic names and frame ids.
_SDF_TOPIC_TAGS = ('topic', 'odom_topic', 'tf_topic')
_SDF_FRAME_TAGS = ('frame_id', 'child_frame_id', 'gz_frame_id')

# (name, ros_type, gz_type, direction) of the per-robot bridge entries. The
# global ``clock`` bridge is launched separately and is not listed here.
BRIDGED_TOPICS: list[tuple[str, str, str, str]] = [
    ('joint_states', 'sensor_msgs/msg/JointState', 'gz.msgs.Model', 'GZ_TO_ROS'),
    ('odom', 'nav_msgs/msg/Odometry', 'gz.msgs.Odometry', 'GZ_TO_ROS'),
    ('tf', 'tf2_msgs/msg/TFMessage', 'gz.msgs.Pose_V', 'GZ_TO_ROS'),
    ('cmd_vel', 'geometry_msgs/msg/TwistStamped', 'gz.msgs.Twist', 'ROS_TO_GZ'),
    ('imu', 'sensor_msgs/msg/Imu', 'gz.msgs.IMU', 'GZ_TO_ROS'),
    ('scan', 'sensor_msgs/msg/LaserScan', 'gz.msgs.LaserScan', 'GZ_TO_ROS'),
]

# Cartographer Lua options that name a frame (quoted string values).
CARTOGRAPHER_FRAME_OPTIONS: tuple[str, ...] = (
    'map_frame', 'tracking_frame', 'published_frame', 'odom_frame',
)


def frame(ns: str, name: str) -> str:
    """Return the namespaced TF frame id for a frame name, e.g. r1/base_link."""
    return f'{ns}/{name}'


def topic(ns: str, name: str) -> str:
    """Return the absolute namespaced topic for a topic name, e.g. /r1/scan."""
    return f'/{ns}/{name.lstrip("/")}'


def _sdf_element_pattern(tags: tuple[str, ...]) -> re.Pattern[str]:
    """Regex matching ``<tag>text</tag>`` for any of ``tags``, keeping whitespace."""
    alternatives = '|'.join(re.escape(tag) for tag in tags)
    return re.compile(
        rf'(?P<open><(?P<tag>{alternatives})>)(?P<lead>\s*)(?P<text>[^<]*?)(?P<trail>\s*)'
        r'(?P<close></(?P=tag)>)'
    )


_SDF_TOPIC_RE = _sdf_element_pattern(_SDF_TOPIC_TAGS)
_SDF_FRAME_RE = _sdf_element_pattern(_SDF_FRAME_TAGS)


def _sub_sdf_element(pattern: re.Pattern[str], new_name: Callable[[str], str], text: str) -> str:
    """Replace the text of every element matched by ``pattern`` with ``new_name(text)``."""
    def _replace(match: re.Match[str]) -> str:
        return (f"{match['open']}{match['lead']}{new_name(match['text'])}"
                f"{match['trail']}{match['close']}")

    return pattern.sub(_replace, text)


def namespace_model_sdf(sdf_text: str, ns: str) -> str:
    """Rewrite the TurtleBot3 Gazebo model SDF so its names live under ``ns``.

    Topic elements (``topic``, ``odom_topic``, ``tf_topic``) become
    ``/<ns>/<name>``; frame elements (``frame_id``, ``child_frame_id``,
    ``gz_frame_id``) become ``<ns>/<name>``. Everything else is left unchanged.
    Raises a ValueError if the result has no ``<tf_topic>``, i.e. the input
    does not look like the expected TurtleBot3 model.
    """
    result = _sub_sdf_element(_SDF_TOPIC_RE, lambda name: topic(ns, name), sdf_text)
    result = _sub_sdf_element(_SDF_FRAME_RE, lambda name: frame(ns, name), result)
    if '<tf_topic>' not in result:
        raise ValueError('SDF has no <tf_topic> element; not a TurtleBot3 model SDF')
    return result


def bridge_config(ns: str) -> list[dict[str, str]]:
    """Return the ros_gz_bridge entries for one robot, all under ``/<ns>/``."""
    return [
        {
            'ros_topic_name': topic(ns, name),
            'gz_topic_name': topic(ns, name),
            'ros_type_name': ros_type,
            'gz_type_name': gz_type,
            'direction': direction,
        }
        for name, ros_type, gz_type, direction in BRIDGED_TOPICS
    ]


def namespace_cartographer_lua(lua_text: str, ns: str) -> str:
    """Prefix the frame options of a Cartographer Lua config with ``<ns>/``.

    Raises a ValueError if any of CARTOGRAPHER_FRAME_OPTIONS is missing.
    """
    missing = [
        option for option in CARTOGRAPHER_FRAME_OPTIONS
        if not re.search(rf'\b{option}\s*=\s*"', lua_text)
    ]
    if missing:
        raise ValueError(f'Cartographer Lua is missing frame option(s): {", ".join(missing)}')

    options = '|'.join(CARTOGRAPHER_FRAME_OPTIONS)
    pattern = re.compile(rf'(\b(?:{options})\s*=\s*")([^"]*)(")')
    return pattern.sub(lambda m: f'{m.group(1)}{frame(ns, m.group(2))}{m.group(3)}', lua_text)


def _namespace_nav2_node(node: object, ns: str) -> None:
    """Rewrite frame and topic values in ``node`` in place (recursively)."""
    if isinstance(node, dict):
        for key, value in node.items():
            if isinstance(key, str) and isinstance(value, str) and value:
                if key.endswith(('frame', 'frame_id')) \
                        and not value.startswith(f'{ns}/'):
                    node[key] = frame(ns, value)
                    continue
                if key.endswith('topic') and value.startswith('/') \
                        and not value.startswith(f'/{ns}/'):
                    node[key] = topic(ns, value)
                    continue
            _namespace_nav2_node(value, ns)
    elif isinstance(node, list):
        for item in node:
            _namespace_nav2_node(item, ns)


def use_merged_map_for_global_planning(params: dict, merged_frame: str) -> None:
    """Point the global costmap and ``bt_navigator`` at the merged map, in place.

    ``StaticLayer`` of a non-rolling costmap copies the incoming grid 1:1 into
    the costmap and does not transform between frames, so the costmap's
    ``global_frame`` must be the merged map's own frame. The robot's TF tree
    then needs ``merged_frame -> <ns>/map`` (see ``sim_slam.launch.py``).
    ``map_topic`` is absolute so it is not resolved under the robot namespace.
    """
    costmap = params['global_costmap']['global_costmap']['ros__parameters']
    costmap['global_frame'] = merged_frame
    costmap.setdefault('static_layer', {})['map_topic'] = f'/{merged_frame}'
    params['bt_navigator']['ros__parameters']['global_frame'] = merged_frame


def namespace_nav2_params(params: dict, ns: str, merged_frame: str | None = None) -> dict:
    """Return a copy of parsed nav2 params with frames and absolute topics under ``ns``.

    Frame values (keys ending in ``frame``/``frame_id``, or exactly ``frame``)
    get ``<ns>/`` prefixed. Absolute topic values (keys ending in ``topic``
    starting with ``/``) become ``/<ns>/...``. Relative topics stay unchanged.
    With ``merged_frame`` the global costmap and ``bt_navigator`` plan on the
    merged map instead (see ``use_merged_map_for_global_planning``), so a robot
    can navigate into areas only its teammate has mapped. The input is not
    mutated, and applying the function twice gives the same result as once.
    """
    rewritten = copy.deepcopy(params)
    _namespace_nav2_node(rewritten, ns)
    if merged_frame is not None:
        use_merged_map_for_global_planning(rewritten, merged_frame)
    return rewritten
