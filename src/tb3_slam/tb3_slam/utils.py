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

# Distance from the inner face of the south wall to the robot spawn. The robot
# starts near the wall (edge), facing into the room, to mimic a real-world
# "robot enters the room" scenario. Must leave enough clearance for the robot
# footprint plus Nav2 inflation so the spawn is never inside an (inflated) wall.
SPAWN_WALL_MARGIN: float = 0.7

# Edge length of the square world in metres. Only the two study sizes are
# valid (8 x 8 m and 12 x 12 m), selected per world via the world_size
# argument (launch arg / training_runner --world-size).
WORLD_SIZES: Tuple[float, ...] = (8.0, 12.0)
DEFAULT_WORLD_SIZE: float = WORLD_SIZES[0]
WALL_THICKNESS: float = 0.15

# House-like room geometry for the BSP generator. Rooms are split until they
# are at most MAX_ROOM_AREA (bedroom .. living room); a cut never leaves a room
# smaller than MIN_ROOM_AREA or narrower than MIN_ROOM_SPAN.
MIN_ROOM_AREA: float = 9.0    # m²
MAX_ROOM_AREA: float = 20.0   # m²
MIN_ROOM_SPAN: float = 2.0    # m
MAX_ROOM_ASPECT: float = 2.5  # long side / short side
# Cut position as a fraction of the room's long side (near-balanced cuts).
SPLIT_FRACTION: Tuple[float, float] = (0.35, 0.65)
# Minimum wall piece next to a door: a door is either flush with a corner or
# has at least this much wall on each side, and later walls keep this distance
# from existing doors. Avoids short wall stubs, which look unrealistic and are
# Nav2 trouble spots.
MIN_DOOR_SIDE_WALL: float = 0.8


def world_meta_path(world_file: str) -> Path:
    """Sidecar metadata path for a given world SDF.

    Metadata is stored next to the world (``<world>.meta.json``) so a saved
    world carries its own coverage denominator and spawn pose, instead of
    relying on the shared /tmp file which only describes the *last generated*
    world.
    """
    return Path(world_file).with_suffix('.meta.json')


def write_world_meta(path: Path, expected_traversable_cells: int,
                     spawn_x: float, spawn_y: float, world_size: float) -> None:
    """Write world metadata (traversable cells, spawn pose, size) as JSON."""
    path.write_text(
        json.dumps({
            'expected_traversable_cells': expected_traversable_cells,
            'spawn_x': spawn_x,
            'spawn_y': spawn_y,
            'world_size': world_size,
        })
    )


def spawn_pose(world_size: float, wall_thickness: float) -> Tuple[float, float]:
    """Robot spawn (x, y): centred in x, just inside the south wall.

    Derived from world_size so it stays at the edge when worlds grow larger.
    """
    half = world_size / 2.0
    y = -half + wall_thickness / 2.0 + SPAWN_WALL_MARGIN
    return (0.0, y)


def check_world_size(world_size: float) -> float:
    """Return world_size if it is one of WORLD_SIZES, else raise ValueError."""
    if world_size not in WORLD_SIZES:
        valid = ', '.join(f'{s:g}' for s in WORLD_SIZES)
        raise ValueError(f'world_size must be one of {valid} (got {world_size:g})')
    return world_size


def generate_random_world(
    file_path: str, world_size: float = DEFAULT_WORLD_SIZE,
) -> tuple[bool, int]:
    """
    Generates a closed square Gazebo SDF world (world_size x world_size m)
    with house-like interior rooms; the number of rooms follows the area.
    Writes the world to file_path and writes expected traversable cell count
    (plus spawn pose) both to a sidecar next to the world (``<world>.meta.json``,
    so a saved world can be reused via --world with the correct coverage
    denominator) and to the shared WORLD_META_FILE that running nodes read.

    Returns:
        (success, expected_traversable_cells)
        expected_traversable_cells is 0 on failure.

    Raises:
        ValueError: world_size is not one of WORLD_SIZES.
    """
    check_world_size(world_size)

    try:
        wall_height = 1.2
        wall_thickness = WALL_THICKNESS
        # Robot-scale clearance around the spawn, independent of world size.
        keep_out_half = 0.5

        half = world_size / 2.0

        # Robot spawns near the south wall (edge entry), not the centre.
        spawn_x, spawn_y = spawn_pose(world_size, wall_thickness)

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
                max_x <= spawn_x - keep_out_half
                or min_x >= spawn_x + keep_out_half
                or max_y <= spawn_y - keep_out_half
                or min_y >= spawn_y + keep_out_half
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
        # Rooms are split recursively until they are house-sized: a room is
        # split while it is larger than MAX_ROOM_AREA or more elongated than
        # MAX_ROOM_ASPECT (stop criterion instead of a fixed split count, so
        # the room count follows the world area). Each cut runs across the
        # room's long axis near its middle (kd-tree style), which avoids
        # slivers and corridor-like strips. Every split wall gets one door, so
        # all rooms stay connected. Room bounds are wall centre lines.
        door_width = MIN_PATH_WIDTH

        # Door openings created by previous splits, stored as axis-aligned
        # free rectangles (xmin, xmax, ymin, ymax), widened by
        # MIN_DOOR_SIDE_WALL along the opening. A later perpendicular split
        # wall overlapping the opening itself would narrow the door below
        # door_width (room unreachable); one landing just beside it would leave
        # a short wall stub. Both are rejected (see blocks_existing_door).
        doors: List[Tuple[float, float, float, float]] = []

        def blocks_existing_door(segments) -> bool:
            """True if any wall segment overlaps or crowds an existing door.

            segments are (center_x, center_y, size_x, size_y) boxes, doors the
            widened opening rectangles described above.
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

        def needs_split(xmin: float, xmax: float, ymin: float, ymax: float) -> bool:
            w, h = xmax - xmin, ymax - ymin
            return w * h > MAX_ROOM_AREA or max(w, h) > MAX_ROOM_ASPECT * min(w, h)

        def door_interval(lo: float, hi: float) -> Tuple[float, float]:
            """Door opening [d0, d1] along a wall whose free span is [lo, hi].

            Either flush with a corner (no wall stub on that side) or with at
            least MIN_DOOR_SIDE_WALL of wall on both sides, like real doors.
            """
            if hi - lo >= door_width + 2 * MIN_DOOR_SIDE_WALL and random.random() < 0.5:
                d0 = random.uniform(lo + MIN_DOOR_SIDE_WALL,
                                    hi - MIN_DOOR_SIDE_WALL - door_width)
                return d0, d0 + door_width
            if random.random() < 0.5:
                return lo, lo + door_width
            return hi - door_width, hi

        rooms: List[Tuple[float, float, float, float]] = []
        pending: List[Tuple[float, float, float, float]] = [(-half, half, -half, half)]
        split_count = 0

        while pending:
            xmin, xmax, ymin, ymax = pending.pop()
            if not needs_split(xmin, xmax, ymin, ymax):
                rooms.append((xmin, xmax, ymin, ymax))
                continue

            # Cut across the long axis: a vertical wall splits the x span.
            vertical = (xmax - xmin) >= (ymax - ymin)
            lo, hi = (xmin, xmax) if vertical else (ymin, ymax)
            along_lo, along_hi = (ymin, ymax) if vertical else (xmin, xmax)
            span, along = hi - lo, along_hi - along_lo

            split_success = False
            for _ in range(40):
                cut = lo + span * random.uniform(*SPLIT_FRACTION)
                side = min(cut - lo, hi - cut)
                if side < MIN_ROOM_SPAN or side * along < MIN_ROOM_AREA:
                    continue

                # Door is placed on the free span between the bounding walls.
                d0, d1 = door_interval(along_lo + wall_thickness / 2.0,
                                       along_hi - wall_thickness / 2.0)
                segments = []
                for s, e in ((along_lo, d0), (d1, along_hi)):
                    # A flush door leaves only a piece hidden inside the
                    # perpendicular wall – skip it.
                    if e - s <= wall_thickness / 2.0 + 1e-6:
                        continue
                    if vertical:
                        segments.append((cut, (s + e) / 2.0, wall_thickness, e - s))
                    else:
                        segments.append(((s + e) / 2.0, cut, e - s, wall_thickness))

                if any(intersects_keep_out(x, y, sx, sy) for x, y, sx, sy in segments):
                    continue
                if blocks_existing_door(segments):
                    continue

                axis = 'v' if vertical else 'h'
                for seg_idx, (x, y, sx, sy) in enumerate(segments):
                    name = f"split_{split_count}_{axis}_{seg_idx}"
                    walls.append(create_wall(name, x, y, sx, sy))

                m = MIN_DOOR_SIDE_WALL
                if vertical:
                    doors.append((cut - wall_thickness / 2.0, cut + wall_thickness / 2.0,
                                  d0 - m, d1 + m))
                    pending.append((xmin, cut, ymin, ymax))
                    pending.append((cut, xmax, ymin, ymax))
                else:
                    doors.append((d0 - m, d1 + m,
                                  cut - wall_thickness / 2.0, cut + wall_thickness / 2.0))
                    pending.append((xmin, xmax, ymin, cut))
                    pending.append((xmin, xmax, cut, ymax))
                split_count += 1
                split_success = True
                break

            if not split_success:
                # Cannot be split validly (spawn keep-out, door clash or too
                # small to halve) – keep it as one larger room.
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

        # Write metadata both next to the world (sidecar, so a saved world
        # carries its own coverage denominator for --world reuse) and to the
        # shared /tmp file that running nodes read by default.
        write_world_meta(
            world_meta_path(file_path),
            expected_traversable_cells, spawn_x, spawn_y, world_size,
        )
        write_world_meta(
            WORLD_META_FILE, expected_traversable_cells, spawn_x, spawn_y,
            world_size,
        )

        return True, expected_traversable_cells

    except Exception:
        return False, 0
