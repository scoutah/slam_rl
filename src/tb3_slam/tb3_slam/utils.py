import random
import os
from typing import List


def generate_random_world(file_path: str) -> bool:
    """
    Generates a closed Gazebo SDF world (~5x5m) with room-like interior walls.
    Writes the world to file_path.
    Returns True on success, False otherwise.
    """

    try:
        world_size = 5.0
        wall_height = 1.2
        wall_thickness = 0.15

        half = world_size / 2.0

        def create_wall(name: str, x: float, y: float,
                        size_x: float, size_y: float) -> str:
            return f"""
    <model name="{name}">
      <static>true</static>
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

        # --------------------------------------------------
        # 1. Outer boundary (closed box room)
        # --------------------------------------------------

        walls.append(create_wall("outer_north", 0,  half, world_size, wall_thickness))
        walls.append(create_wall("outer_south", 0, -half, world_size, wall_thickness))
        walls.append(create_wall("outer_east",  half, 0, wall_thickness, world_size))
        walls.append(create_wall("outer_west", -half, 0, wall_thickness, world_size))

        # --------------------------------------------------
        # 2. Interior room-like structure
        # --------------------------------------------------

        num_divisions = random.randint(2, 3)
        door_width = 0.8

        for i in range(num_divisions):

            horizontal = random.random() < 0.5

            if horizontal:
                y = random.uniform(-half * 0.6, half * 0.6)

                door_center = random.uniform(-half * 0.5, half * 0.5)

                left_length = (door_center - door_width/2) - (-half)
                right_length = (half) - (door_center + door_width/2)

                if left_length > 0.3:
                    walls.append(
                        create_wall(
                            f"hwall_{i}_left",
                            -half + left_length/2,
                            y,
                            left_length,
                            wall_thickness
                        )
                    )

                if right_length > 0.3:
                    walls.append(
                        create_wall(
                            f"hwall_{i}_right",
                            door_center + door_width/2 + right_length/2,
                            y,
                            right_length,
                            wall_thickness
                        )
                    )

            else:
                x = random.uniform(-half * 0.6, half * 0.6)

                door_center = random.uniform(-half * 0.5, half * 0.5)

                bottom_length = (door_center - door_width/2) - (-half)
                top_length = (half) - (door_center + door_width/2)

                if bottom_length > 0.3:
                    walls.append(
                        create_wall(
                            f"vwall_{i}_bottom",
                            x,
                            -half + bottom_length/2,
                            wall_thickness,
                            bottom_length
                        )
                    )

                if top_length > 0.3:
                    walls.append(
                        create_wall(
                            f"vwall_{i}_top",
                            x,
                            door_center + door_width/2 + top_length/2,
                            wall_thickness,
                            top_length
                        )
                    )

        # --------------------------------------------------
        # 3. Assemble SDF
        # --------------------------------------------------

        sdf_content = f"""<?xml version="1.0"?>
<sdf version="1.6">
  <world name="default">

  <plugin filename="gz-sim-physics-system" name="gz::sim::systems::Physics"/>
  <plugin filename="gz-sim-user-commands-system" name="gz::sim::systems::UserCommands"/>
  <plugin filename="gz-sim-scene-broadcaster-system" name="gz::sim::systems::SceneBroadcaster"/>
  <plugin filename="gz-sim-sensors-system" name="gz::sim::systems::Sensors">
    <render_engine>ogre2</render_engine>
  </plugin>
  <plugin filename="gz-sim-imu-system" name="gz::sim::systems::Imu"/>

  <include>
    <uri>https://fuel.gazebosim.org/1.0/OpenRobotics/models/Ground Plane</uri>
  </include>

  <include>
    <uri>https://fuel.gazebosim.org/1.0/OpenRobotics/models/Sun</uri>
  </include>

  <scene>
    <shadows>false</shadows>
  </scene>

  <physics type="ode">
    <real_time_update_rate>1000.0</real_time_update_rate>
    <max_step_size>0.001</max_step_size>
    <real_time_factor>1</real_time_factor>
  </physics>

  <!-- Walls -->
  {''.join(walls)}

  </world>
</sdf>
"""

        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(sdf_content)

        return True

    except Exception:
        return False
