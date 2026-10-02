import json
import random
import os
from pathlib import Path
from typing import List, Tuple

WORLD_META_FILE = Path('/tmp/slam_world_meta.json')

# Must match the SLAM/Cartographer map resolution
MAP_RESOLUTION: float = 0.05  # metres per cell

# Minimum free-path width (door gap + corridor). TB3 Burger footprint is ~0.22 m
# but Nav2's default inflation radius of ~0.55 m needs > 2x inflation for the
# planner to route through a passage, i.e. > 1.1 m. Anything smaller produces
# rooms the robot can see but cannot enter.
MIN_PATH_WIDTH: float = 1.1


def generate_random_world(file_path: str) -> tuple[bool, int]:
    """
    Generates a closed Gazebo SDF world (~5x5m) with room-like interior walls.
    Writes the world to file_path and writes expected traversable cell count
    to WORLD_META_FILE so exploration nodes can use it as a coverage target.

    Returns:
        (success, expected_traversable_cells)
        expected_traversable_cells is 0 on failure.
    """

    try:
        world_size = 5.0
        wall_height = 1.2
        wall_thickness = 0.15
        keep_out_half = 0.5

        half = world_size / 2.0

        # Track footprint area of every wall to derive free-space estimate
        _wall_areas: List[float] = []

        def create_wall(name: str, x: float, y: float,
                        size_x: float, size_y: float) -> str:
            _wall_areas.append(size_x * size_y)
            return f"""
    <model name="{name}">
      <static>1</static>
      <pose>{x} {y} {wall_height/2} 0 0 0</pose>
      <link name="link">
        <collision name="collision">
          <geometry>
            <box>
              <size>{size_x} {size_y} {wall_height}</size>
            </box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box>
              <size>{size_x} {size_y} {wall_height}</size>
            </box>
          </geometry>
        </visual>
      </link>
    </model>
"""

        walls: List[str] = []

        def intersects_keep_out(
            center_x: float,
            center_y: float,
            size_x: float,
            size_y: float,
        ) -> bool:
            half_x = size_x / 2.0
            half_y = size_y / 2.0

            min_x = center_x - half_x
            max_x = center_x + half_x
            min_y = center_y - half_y
            max_y = center_y + half_y

            return not (
                max_x <= -keep_out_half
                or min_x >= keep_out_half
                or max_y <= -keep_out_half
                or min_y >= keep_out_half
            )

        # --------------------------------------------------
        # 1. Outer boundary (closed box room)
        # --------------------------------------------------

        walls.append(create_wall("outer_north", 0,  half, world_size, wall_thickness))
        walls.append(create_wall("outer_south", 0, -half, world_size, wall_thickness))
        walls.append(create_wall("outer_east",  half, 0, wall_thickness, world_size))
        walls.append(create_wall("outer_west", -half, 0, wall_thickness, world_size))

        # --------------------------------------------------
        # 2. Interior room-like structure (connected BSP split)
        # --------------------------------------------------

        rooms: List[Tuple[float, float, float, float]] = [(-half, half, -half, half)]

        target_splits = random.randint(3, 5)
        # Rooms must be wide enough to (a) fit a door of MIN_PATH_WIDTH and
        # (b) still leave clearance on either side of the door. Buffer of
        # 0.4 m keeps approach corridors comfortably wider than the door.
        min_room_span = max(1.4, MIN_PATH_WIDTH + 0.4)
        door_width = MIN_PATH_WIDTH
        # Any wall sub-segment shorter than this collapses into the door, so
        # the actual opening can only ever be ≥ door_width.
        min_segment = 0.25

        # Door openings created by previous splits, stored as axis-aligned
        # free rectangles (xmin, xmax, ymin, ymax). A later perpendicular split
        # wall whose solid segment overlaps one of these openings would cut the
        # door into a passage narrower than door_width, making a room
        # unreachable. We reject such splits (see blocks_existing_door).
        doors: List[Tuple[float, float, float, float]] = []

        def blocks_existing_door(segments) -> bool:
            """True if any wall segment overlaps an existing door opening.

            segments are (center_x, center_y, size_x, size_y) boxes. A door is
            the free rectangle left in a parent wall; if a new perpendicular
            wall intrudes into that rectangle it narrows the opening below
            door_width, so the split must be rejected.
            """
            for cx, cy, sx, sy in segments:
                seg_xmin = cx - sx / 2.0
                seg_xmax = cx + sx / 2.0
                seg_ymin = cy - sy / 2.0
                seg_ymax = cy + sy / 2.0
                for dxmin, dxmax, dymin, dymax in doors:
                    # Strict overlap of the segment box with the open door
                    # rectangle (touching at an edge is allowed).
                    if (seg_xmin < dxmax and seg_xmax > dxmin
                            and seg_ymin < dymax and seg_ymax > dymin):
                        return True
            return False

        split_count = 0
        attempts = 0
        max_attempts = 200

        while split_count < target_splits and attempts < max_attempts:
            attempts += 1

            splittable = []
            for idx, (xmin, xmax, ymin, ymax) in enumerate(rooms):
                span_x = xmax - xmin
                span_y = ymax - ymin

                can_vertical = span_x >= (2 * min_room_span + wall_thickness)
                can_horizontal = span_y >= (2 * min_room_span + wall_thickness)

                if can_vertical or can_horizontal:
                    splittable.append((idx, can_vertical, can_horizontal))

            if not splittable:
                break

            room_idx, can_vertical, can_horizontal = random.choice(splittable)
            xmin, xmax, ymin, ymax = rooms.pop(room_idx)

            if can_vertical and can_horizontal:
                orientation = random.choice(["vertical", "horizontal"])
            elif can_vertical:
                orientation = "vertical"
            else:
                orientation = "horizontal"

            split_success = False

            for _ in range(40):
                if orientation == "vertical":
                    split_x = random.uniform(xmin + min_room_span, xmax - min_room_span)

                    door_center = random.uniform(
                        ymin + door_width / 2.0,
                        ymax - door_width / 2.0,
                    )

                    bottom_len = (door_center - door_width / 2.0) - ymin
                    top_len = ymax - (door_center + door_width / 2.0)

                    segments = []
                    if bottom_len >= min_segment:
                        segments.append((split_x, ymin + bottom_len / 2.0, wall_thickness, bottom_len))
                    if top_len >= min_segment:
                        segments.append((split_x, door_center + door_width / 2.0 + top_len / 2.0, wall_thickness, top_len))

                    if not segments:
                        continue

                    if any(intersects_keep_out(x, y, sx, sy) for x, y, sx, sy in segments):
                        continue

                    if blocks_existing_door(segments):
                        continue

                    for seg_idx, (x, y, sx, sy) in enumerate(segments):
                        walls.append(create_wall(f"split_{split_count}_v_{seg_idx}", x, y, sx, sy))

                    # Record the door opening: free gap in y at x = split_x.
                    doors.append((
                        split_x - wall_thickness / 2.0,
                        split_x + wall_thickness / 2.0,
                        door_center - door_width / 2.0,
                        door_center + door_width / 2.0,
                    ))

                    rooms.append((xmin, split_x, ymin, ymax))
                    rooms.append((split_x, xmax, ymin, ymax))
                    split_success = True

                else:
                    split_y = random.uniform(ymin + min_room_span, ymax - min_room_span)

                    door_center = random.uniform(
                        xmin + door_width / 2.0,
                        xmax - door_width / 2.0,
                    )

                    left_len = (door_center - door_width / 2.0) - xmin
                    right_len = xmax - (door_center + door_width / 2.0)

                    segments = []
                    if left_len >= min_segment:
                        segments.append((xmin + left_len / 2.0, split_y, left_len, wall_thickness))
                    if right_len >= min_segment:
                        segments.append((door_center + door_width / 2.0 + right_len / 2.0, split_y, right_len, wall_thickness))

                    if not segments:
                        continue

                    if any(intersects_keep_out(x, y, sx, sy) for x, y, sx, sy in segments):
                        continue

                    if blocks_existing_door(segments):
                        continue

                    for seg_idx, (x, y, sx, sy) in enumerate(segments):
                        walls.append(create_wall(f"split_{split_count}_h_{seg_idx}", x, y, sx, sy))

                    # Record the door opening: free gap in x at y = split_y.
                    doors.append((
                        door_center - door_width / 2.0,
                        door_center + door_width / 2.0,
                        split_y - wall_thickness / 2.0,
                        split_y + wall_thickness / 2.0,
                    ))

                    rooms.append((xmin, xmax, ymin, split_y))
                    rooms.append((xmin, xmax, split_y, ymax))
                    split_success = True

                if split_success:
                    split_count += 1
                    break

            if not split_success:
                rooms.append((xmin, xmax, ymin, ymax))

        # --------------------------------------------------
        # 3. Assemble SDF
        # --------------------------------------------------

        sdf_content = f"""<?xml version="1.0"?>
<sdf version="1.8">
  <world name="default">
    <physics type="ode">
      <real_time_update_rate>1000.0</real_time_update_rate>
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1</real_time_factor>
      <ode>
        <solver>
          <type>quick</type>
          <iters>150</iters>
          <precon_iters>0</precon_iters>
          <sor>1.400000</sor>
          <use_dynamic_moi_rescaling>1</use_dynamic_moi_rescaling>
        </solver>
        <constraints>
          <cfm>0.00001</cfm>
          <erp>0.2</erp>
          <contact_max_correcting_vel>2000.000000</contact_max_correcting_vel>
          <contact_surface_layer>0.01000</contact_surface_layer>
        </constraints>
      </ode>
    </physics>
    <plugin
      filename="gz-sim-physics-system"
      name="gz::sim::systems::Physics">
    </plugin>
    <plugin
      filename="gz-sim-user-commands-system"
      name="gz::sim::systems::UserCommands">
    </plugin>
    <plugin
      filename="gz-sim-scene-broadcaster-system"
      name="gz::sim::systems::SceneBroadcaster">
    </plugin>
    <plugin
      filename="gz-sim-sensors-system"
      name="gz::sim::systems::Sensors">
      <render_engine>ogre2</render_engine>
    </plugin>
    <plugin
      filename="gz-sim-imu-system"
      name="gz::sim::systems::Imu">
    </plugin>

    <include>
      <uri>https://fuel.gazebosim.org/1.0/OpenRobotics/models/Ground Plane</uri>
    </include>

    <include>
      <uri>https://fuel.gazebosim.org/1.0/OpenRobotics/models/Sun</uri>
    </include>

  <!-- Walls -->
  {''.join(walls)}

  </world>
</sdf>
"""

        # -------------------------------------------------------
        # 4. Calculate expected traversable cells
        # -------------------------------------------------------
        # Outer walls overlap at the four corners, so subtract those.
        total_wall_area = sum(_wall_areas) - 4 * (wall_thickness ** 2)
        free_area = max(0.0, world_size ** 2 - total_wall_area)
        expected_traversable_cells = int(free_area / MAP_RESOLUTION ** 2)

        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)

        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(sdf_content)

        WORLD_META_FILE.write_text(
            json.dumps({'expected_traversable_cells': expected_traversable_cells})
        )

        return True, expected_traversable_cells

    except Exception:
        return False, 0
