"""Relay Nav2's Twist cmd_vel to TwistStamped for the Jazzy gz bridge, with a scan-based stop."""
import math

from geometry_msgs.msg import Twist, TwistStamped
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan


def clearance_ahead(
    ranges: list[float], angle_min: float, angle_increment: float,
    direction: float, half_width: float, range_min: float, range_max: float,
) -> float | None:
    """Return the distance to the nearest return inside the driving corridor.

    The corridor is ``2 * half_width`` wide and extends from the sensor along the
    direction of travel (``+1`` forward, ``-1`` backward). Teammates show up in
    the scan like any other obstacle. ``None`` means nothing within ``range_max``.
    """
    nearest: float | None = None
    for i, r in enumerate(ranges):
        if not math.isfinite(r) or r < range_min or r > range_max:
            continue
        a = angle_min + i * angle_increment
        along = direction * r * math.cos(a)
        if along <= 0.0 or abs(r * math.sin(a)) > half_width:
            continue
        if nearest is None or along < nearest:
            nearest = along
    return nearest


def limit_speed(v: float, clearance: float | None, stop_dist: float, decel: float) -> float:
    """Cap speed ``v`` (m/s, signed) so the robot can still stop ``stop_dist`` short.

    Allowed speed is ``sqrt(2 * decel * (clearance - stop_dist))``: zero at or
    inside ``stop_dist``, unlimited when the corridor is clear.
    """
    if clearance is None or v == 0.0:
        return v
    free = clearance - stop_dist
    allowed = math.sqrt(2.0 * decel * free) if free > 0.0 else 0.0
    return math.copysign(min(abs(v), allowed), v)


class CmdVelRelay(Node):
    """Forward velocity commands, capped so the robot never drives into an obstacle.

    Nav2's local costmap is the first line of defence; this guard sits after it
    (and after the velocity smoother) and works straight off the lidar, so it also
    stops a robot that is about to hit its teammate or a wall the costmap missed.
    Angular velocity is left alone so a blocked robot can still turn away.
    """

    def __init__(self) -> None:
        super().__init__('cmd_vel_relay')

        self.declare_parameter('input_topic', '/cmd_vel_smoothed')
        self.declare_parameter('output_topic', '/cmd_vel')
        self.declare_parameter('frame_id', 'base_link')
        self.declare_parameter('scan_topic', '')           # empty disables the guard
        self.declare_parameter('stop_distance', 0.20)      # scan origin to obstacle, m
        self.declare_parameter('corridor_half_width', 0.17)
        self.declare_parameter('decel', 0.5)               # m/s^2 used for the speed cap
        self.declare_parameter('scan_timeout', 1.0)        # s; stale scan => stop driving

        input_topic = self.get_parameter('input_topic').get_parameter_value().string_value
        output_topic = self.get_parameter('output_topic').get_parameter_value().string_value
        scan_topic = self.get_parameter('scan_topic').get_parameter_value().string_value
        self._frame_id = self.get_parameter('frame_id').get_parameter_value().string_value
        self._stop = self.get_parameter('stop_distance').value
        self._half_width = self.get_parameter('corridor_half_width').value
        self._decel = self.get_parameter('decel').value
        self._scan_timeout = self.get_parameter('scan_timeout').value

        self._scan: LaserScan | None = None
        self._scan_time = 0.0
        self._guard = bool(scan_topic)
        self._blocked_logged = False

        self._publisher = self.create_publisher(TwistStamped, output_topic, 10)
        self._subscription = self.create_subscription(
            Twist,
            input_topic,
            self._on_cmd_vel,
            10
        )
        if self._guard:
            self.create_subscription(
                LaserScan, scan_topic, self._on_scan, qos_profile_sensor_data)

        self.get_logger().info(
            f'Relaying Twist from {input_topic} to TwistStamped on {output_topic}'
            + (f' (collision guard on {scan_topic})' if self._guard else '')
        )

    def _now(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def _on_scan(self, msg: LaserScan) -> None:
        self._scan = msg
        self._scan_time = self._now()

    def _guarded(self, msg: Twist) -> Twist:
        """Return ``msg`` with the linear speed capped by the lidar clearance."""
        vx = msg.linear.x
        if not self._guard or vx == 0.0:
            return msg
        scan = self._scan
        if scan is None or self._now() - self._scan_time > self._scan_timeout:
            # No fresh view of the surroundings: do not drive blind (fail safe).
            vx_new = 0.0
        else:
            clear = clearance_ahead(
                list(scan.ranges), scan.angle_min, scan.angle_increment,
                1.0 if vx > 0.0 else -1.0, self._half_width,
                scan.range_min, scan.range_max,
            )
            vx_new = limit_speed(vx, clear, self._stop, self._decel)
        if vx_new != vx:
            if not self._blocked_logged and vx_new == 0.0:
                self.get_logger().warning(
                    'collision guard: obstacle ahead, stopping', throttle_duration_sec=2.0)
            self._blocked_logged = vx_new == 0.0
            out = Twist()
            out.linear.x = vx_new
            out.linear.y = msg.linear.y
            out.linear.z = msg.linear.z
            out.angular = msg.angular
            return out
        self._blocked_logged = False
        return msg

    def _on_cmd_vel(self, msg: Twist) -> None:
        stamped = TwistStamped()
        stamped.header.stamp = self.get_clock().now().to_msg()
        stamped.header.frame_id = self._frame_id
        stamped.twist = self._guarded(msg)
        self._publisher.publish(stamped)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = CmdVelRelay()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
