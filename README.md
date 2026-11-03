# SLAM RL

TurtleBot3-basiertes SLAM-Projekt: zwei namespaced Burger-Roboter erkunden
zufällig generierte, zimmerartige Welten in Gazebo. Jeder Roboter fährt eigenes
Cartographer + Nav2; die lokalen Karten werden zu einer globalen `/map`
zusammengeführt, auf der Frontier-Erkennung, Coverage und Episodenende laufen.
Ein Explorer wählt Frontier-Ziele – klassisch (Cost-Utility) oder per RL (DQN).

## Pakete

| Paket | Beschreibung |
|---|---|
| `tb3_slam` | Launch, Welten-Generierung, Map-Merging, geteilte Libs, Episodenverwaltung |
| `tb3_exploration` | Klassischer Explorer (Cost-Utility) |
| `tb3_reinforced_exploration` | RL-Explorer (DQN) |

## Installation

Ziel: **Ubuntu 24.04 + ROS 2 Jazzy + Gazebo (gz-sim)**. Läuft nicht auf macOS.

```bash
sudo apt install \
  ros-jazzy-turtlebot3 ros-jazzy-turtlebot3-simulations \
  ros-jazzy-cartographer-ros ros-jazzy-nav2-bringup

export TURTLEBOT3_MODEL=burger            # in ~/.bashrc aufnehmen
source /opt/ros/jazzy/setup.bash          # in ~/.bashrc aufnehmen
```

Bauen (immer aus dem Repo-Root, **nie** aus `src/`); nach jedem Build neu sourcen:

```bash
colcon build && source install/setup.bash
```

Der RL-Explorer braucht PyTorch in einem projektlokalen `.venv` (das Launch-File
injiziert dessen `site-packages` in den `PYTHONPATH`).

## Starten

```bash
# Klassisch (Standard) / RL / nackter Stack (Ziele von außen)
ros2 launch tb3_slam sim_slam.launch.py exploration_mode:=classic
ros2 launch tb3_slam sim_slam.launch.py exploration_mode:=rl
# weitere Args: headless:=true, rviz:=false, world_size:=12

# Episodisches Training (siehe --help für alle Optionen)
python3 training_runner.py --mode classic --episodes 20 --headless --world-size 12
python3 training_runner.py --world worlds/<id>.sdf --episodes 5   # Welt wiederverwenden

# Offline-Check des Weltengenerators (nur numpy, kein ROS)
python3 tools/check_world_connectivity.py
```

Das Episoden-Timeout ist nicht konfigurierbar; es folgt der Weltgröße
(600 s bei 8 × 8 m, 900 s bei 12 × 12 m).

Simulation mit `Ctrl+C` beenden; überlebt Gazebo das, `pkill -9 -f gz`.

## Tests

```bash
colcon build && source install/setup.bash
colcon test --packages-select tb3_slam && colcon test-result --verbose
```

`colcon test` liefert Exit 0 auch bei Fehlern – immer `colcon test-result`
lesen.

## Code map

Index mit einer Zeile pro Datei. **Vor jeder Code-Änderung lesen, nach jeder
Änderung aktualisieren.** Tiefere Begründungen stehen als kurzer Kommentar an der
betreffenden Stelle im Code, nicht hier und nicht in langen Datei-Headern.

### `src/tb3_slam` — geteiltes Paket

| Datei | Inhalt |
|---|---|
| `launch/sim_slam.launch.py` | Timergesteuerter Start des gesamten Stacks: Weltgen → gz → pro Roboter Spawn/Bridge/Relay/RSP/Cartographer/Start-Spin/Nav2 → Map-Merge → Explorer. `exploration_mode:=classic\|rl\|none`. |
| `tb3_slam/utils.py` | `generate_random_world` (BSP-Haus, nur Studiengrößen 8/12), `spawn_poses`/`load_spawns`, Welt-Metadaten-I/O. |
| `tb3_slam/multi_robot.py` | Pure-Python-Namespacing der Upstream-TurtleBot3-Configs (model.sdf / Bridge / Cartographer-lua / Nav2-Params). Unit-getestet, kein ROS. |
| `tb3_slam/map_merging.py` | Pure-numpy-Fusion: feste Ausdehnung im `map`-Frame, zustandsbehaftete Zell-Priorität (WALL > sticky-FREE > UNCERTAIN > UNKNOWN). Erhält die `frontier_utils`-Zellklassen. |
| `tb3_slam/map_merger.py` | ROS-Node um `map_merging`: abonniert `/<ns>/map`, publiziert `/map` mit 2 Hz TRANSIENT_LOCAL. |
| `tb3_slam/frontier_utils.py` | Geteilte Frontier-Erkennung: Zellklassen (0–25 frei, 26–74 unsicher, 75–100 Wand, −1 unbekannt), BFS über freie Zellen, Gruppen-Centroids, footprint-sichere Ziele. Keine ROS-Messages außer OccupancyGrid. |
| `tb3_slam/frontier_viz.py` | RViz-MarkerArray für Frontiers + aktuelles Ziel. Hält ROS-Viz aus `frontier_utils` heraus. |
| `tb3_slam/multi_robot_explorer.py` | Basis-Node + ROS-freie Typen (`Goal`, `RobotState`, `FrontierCandidate`). Besitzt alle ROS-I/O und Ziel-Überwachung; publiziert bei jedem `/map`-Update die Frontier-/Centroid-Marker auf `/frontiers_viz`. Speist seinen TF-Buffer direkt aus `/<ns>/tf[_static]` (plus globales `/tf_static` für `map→<ns>/map`), also unabhängig vom Viz-`tf_relay`. Dispatcht Ziele im robotereigenen `<ns>/map`-Frame (Nav2 kennt den gemergten `map`-Frame nicht). Subklassen überschreiben nur `select_goals(merged_map, [RobotState])`. |
| `tb3_slam/path_cost.py` | ROS-freie Helfer für den Nav2-Pfadkosten-Cache (`cost_key`, `path_length`). |
| `tb3_slam/coverage_monitor.py` | `timeout_for_world_size` (600 s @8 m, 900 s @12 m), Expected-Cells / Coverage. |
| `tb3_slam/episode_manager.py` | Beobachtet die gemergte `/map`; beendet bei Coverage ≥ Schwelle, stuck (<0,5 %/90 s) oder Timeout; schreibt `/tmp/slam_episode_done`. |
| `tb3_slam/cmd_vel_relay.py` | Nav2 `/<ns>/cmd_vel_smoothed` (Twist) → `/<ns>/cmd_vel` (TwistStamped, für die Jazzy-gz-Bridge). |
| `tb3_slam/tf_relay.py` | Reine Visualisierungshilfe: spiegelt `/<ns>/tf` auf globales `/tf`, damit RViz das RobotModel zeigt. Nur mit RViz gestartet; der Explorer hängt nicht mehr davon ab. |

### `src/tb3_exploration` — klassische Methode

| Datei | Inhalt |
|---|---|
| `tb3_exploration/multi_robot_explorer.py` | `ClassicExplorer` (Node `classic_explorer`): Cost-Utility-Allokator. Baseline erbt aktuell das leere `select_goals`. |
| `tb3_exploration/frontier_explorer.py` | Alter Single-Robot-Frontier-Node (nicht mehr gelauncht). Referenz für Ziel-Überwachung/Ranking. |

### `src/tb3_reinforced_exploration` — RL-Methode

| Datei | Inhalt |
|---|---|
| `tb3_reinforced_exploration/multi_robot_explorer.py` | `RLExplorer` (Node `rl_explorer`): zentraler CTDE-Allokator. Überschreibt `select_goals`, weist beiden Robotern je ein distinktes Frontier zu (ein geteiltes Q-Netz, geteilter Replay), Team-Reward (globaler Coverage-Zuwachs), Persistenz über `data_dir`/`$SLAM_RL_DATA_DIR`. |
| `tb3_reinforced_exploration/rl_core.py` | ROS-freier Lernkern: Features (einzeln + team-aware `make_team_features`), ε-greedy-Policy + Decay, Reward, Replay-Buffer mit Persistenz, Q-Netz (konfigurierbare Feature-Dim), `TeamLearner` (CTDE: Parameter-Sharing + geteilter Buffer), Persist-Dir-Auflösung. Torch lazy importiert. Unit-getestet. |
| `tb3_reinforced_exploration/rl_agent.py` | Alter Single-Robot-DQN-Node (nicht mehr gelauncht). Nutzt `rl_core`. |

### Top-Level-Skripte

| Datei | Inhalt |
|---|---|
| `training_runner.py` | Episoden-Loop: Welt generieren/wiederverwenden, Sim starten, `/tmp/slam_episode_done` pollen, Prozessgruppe killen, CSV in `results/` schreiben. |
| `tools/check_world_connectivity.py` | Offline-Check (nur numpy): Freiraum-Konnektivität + Spawn-Erreichbarkeit für beide Studiengrößen. |

### Inter-Prozess-Dateien (`/tmp`)

- `/tmp/slam_world_meta.json` — `expected_traversable_cells`, `spawns`, `world_size`; von `generate_random_world` geschrieben, von `coverage_monitor` gelesen. Beim Wiederverwenden einer Welt wird deren `<world>.meta.json`-Sidecar hierher kopiert.
- `/tmp/slam_episode_done` — Episoden-Endgrund; von `episode_manager` geschrieben, von `training_runner.py` gepollt.
- `results/<dd-mm-yyyy>-<mode>-<runid>.csv` — `;`-getrennt (Spalte `world_size`).
