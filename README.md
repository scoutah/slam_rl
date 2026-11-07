# SLAM RL

Zwei namespaced TurtleBot3-Burger erkunden zufällig generierte, zimmerartige Welten
in Gazebo. Jeder Roboter fährt eigenes Cartographer + Nav2; die lokalen Karten werden
zu einer globalen `/map` zusammengeführt, auf der Frontier-Erkennung, Coverage und
Episodenende laufen. Ein Explorer wählt die Frontier-Ziele – klassisch (Cost-Utility)
oder per RL (QMIX).

## Pakete

| Paket | Beschreibung |
|---|---|
| `tb3_slam` | Launch, Weltengenerator, Map-Merging, Frontier-Pipeline, Explorer-Basisnode, Episodenverwaltung |
| `tb3_exploration` | Klassischer Explorer (Cost-Utility) |
| `tb3_reinforced_exploration` | RL-Explorer (QMIX, CTDE) |

## Installation

Ziel: **Ubuntu 24.04 + ROS 2 Jazzy + Gazebo (gz-sim)**. Läuft nicht auf macOS
(dort: Docker, siehe `docker/README.md`).

```bash
sudo apt install \
  ros-jazzy-turtlebot3 ros-jazzy-turtlebot3-simulations \
  ros-jazzy-cartographer-ros ros-jazzy-nav2-bringup ros-jazzy-ros-gz

export TURTLEBOT3_MODEL=burger            # in ~/.bashrc aufnehmen
source /opt/ros/jazzy/setup.bash          # in ~/.bashrc aufnehmen
```

Bauen immer aus dem Repo-Root (**nie** aus `src/`), danach neu sourcen:

```bash
colcon build && source install/setup.bash
```

Der RL-Explorer braucht PyTorch in einem projektlokalen `.venv`; das Launch-File hängt
dessen `site-packages` an den `PYTHONPATH`.

## Starten

```bash
# exploration_mode: classic (Standard) | rl | none (nackter Stack, Ziele von außen)
ros2 launch tb3_slam sim_slam.launch.py exploration_mode:=classic
# weitere Args: headless:=true, rviz:=false, world_size:=8|12, coverage_threshold:=0.97

# Episodisches Training (alle Optionen: --help)
python3 training_runner.py --mode classic --episodes 20 --headless --randomize-worlds --world-size 12
python3 training_runner.py --mode rl --episodes 50 --headless --sim-speed 4
python3 training_runner.py --world worlds/<id>.sdf --episodes 5     # gespeicherte Welt

python3 tools/check_world_connectivity.py     # Offline-Check des Generators (nur numpy)
```

- **Weltgrößen:** nur 8 und 12 (m Kantenlänge) sind erlaubt. Ohne `--randomize-worlds`
  wird *eine* Welt für alle Episoden wiederverwendet; `--world-size` ist mit `--world` nicht erlaubt.
- **Timeout:** nicht konfigurierbar, folgt der Weltgröße (600 s bei 8 m, 900 s bei 12 m).
  Alle Zeiten (Timeout, Stuck-Fenster, Warm-up, `elapsed_s` im CSV) zählen in **Simulationszeit**.
- **Sim-Speed:** `--sim-speed N` (bzw. Env `SLAM_SIM_SPEED=N` bei `ros2 launch`) setzt den
  Ziel-Realtime-Faktor (1 = Echtzeit, 0 = unbegrenzt) über eine gepatchte Weltkopie
  `/tmp/slam_world_speed.sdf`. Erreicht wird, was die CPU hergibt (`gz topic -e -t /stats`);
  zu hohe Faktoren verschlechtern Odometrie, Scans und Planung.
- **Episodenende:** Coverage ≥ Schwelle (0,97), `stuck` (< 0,5 % Zuwachs in 90 s), `timeout`
  oder `exhausted` (Team 15 Sim-s ohne Ziel). `exhausted`-Welten kopiert der Runner nach
  `worlds/`, damit sie mit `--world` wiederholbar sind.
- Beenden mit `Ctrl+C`; überlebt Gazebo das: `pkill -9 -f gz`.

## Tests

```bash
colcon build && source install/setup.bash
colcon test && colcon test-result --verbose
```

`colcon test` liefert Exit 0 auch bei Fehlern – immer `colcon test-result` lesen.
Enthalten sind auch flake8/pep257.

## Methoden

**Klassisch** (`classic_explorer`): Greedy-Cost-Utility über bestätigte Nav2-Pfade. Der Nutzen
ist die sichtbare unbekannte/unsichere Fläche im Sensorradius (3,5 m; bekannte Wände
verdecken), abzüglich bereits vergebener Frontiers und überlappender Sichtbereiche. Ein
Rivalen-Abschlag (`utility_gamma`, 0,3) bestraft Ziele, die dem Teamkollegen näher liegen.
Laufende Fahrten bleiben stabil, freie Roboter bekommen neue Ziele. Dieselben Parameter auf
8 × 8 und 12 × 12, kein Lernen. Details: `tb3_exploration/cost_utility.py`.

**RL** (`rl_explorer`): QMIX mit zentralem Training und dezentraler Ausführung. Jeder freie
Roboter bewertet seine Frontiers mit einem geteilten Agenten-Netz auf einer lokalen
Beobachtung; die Roboter entscheiden nacheinander und teilen ihre Ziele als Intents
(Kandidaten in der Nähe eines Intents entfallen). Nur das Training nutzt den globalen
Zustand (QMIX-Mixer, Team-Reward = Coverage-Zuwachs pro Sekunde). Alle Features sind
auf die Kartendiagonale normiert, damit ein auf 8 × 8 trainiertes Netz auf 12 × 12 übertragbar
ist. Checkpoint, Replay und ε liegen in `$SLAM_RL_DATA_DIR` (Standard `<ws>/rl_data`);
ohne Checkpoint startet ein frisches Netz.

## Code map

Index mit einer Zeile pro Datei. **Vor jeder Änderung lesen, bei jeder Änderung aktualisieren.**
Begründungen stehen als kurzer Kommentar im Code.

### `src/tb3_slam`

| Datei | Inhalt |
|---|---|
| `launch/sim_slam.launch.py` | Gesamtstack: Welt (generiert oder `SLAM_WORLD_FILE`) → gz-Server (+GUI) + `/clock`-Bridge → pro Roboter parallel Spawn, Bridge, `cmd_vel_relay`, RSP (2 s), Cartographer (5 s), Start-Spin (6 s) → Nav2 **strikt nacheinander** (r1 @6 s, r2 sobald r1s `nav2_gate` endet; parallele Lifecycle-Aufrufe hängen) → Explorer, sobald das letzte `nav2_gate` endet. `episode_manager`, `map_merger` und die Static-TFs `map→<ns>/map` starten sofort. |
| `launch/nav2_slim.launch.py` | Schlanker Nav2-Bringup: nur `controller_server`, `planner_server`, `behavior_server`, `bt_navigator`, `velocity_smoother` sind lifecycle-managed (kein `smoother_server`, `route_server`, `collision_monitor`, `waypoint_follower`, `docking_server`). |
| `rviz/sim_slam.rviz` | RViz-Konfiguration. |
| `tb3_slam/utils.py` | Weltengenerator (`generate_random_world`, BSP-Haus, nur Größen 8/12), `spawn_poses`/`load_spawns`, Metadaten-I/O, Sim-Speed-Patch (`apply_sim_speed`, `speed_up_world`, `sim_speed_from_env`). |
| `tb3_slam/multi_robot.py` | Namespacing der Upstream-Configs (model.sdf, Bridge, Cartographer-lua, Nav2-Params). Globale Costmap und `bt_navigator` planen auf der gemergten `/map` (Frame `map`), Inflation 0,30 m, damit 1,1-m-Türen passierbar bleiben. Ohne ROS, unit-getestet. |
| `tb3_slam/map_merging.py` | Numpy-Fusion auf festem Gitter im Frame `map`, Zell-Priorität WALL > sticky-FREE > UNCERTAIN > UNKNOWN; `clear_robot_footprints` gibt die 0,22-m-Scheibe jedes Roboters frei (Teamkollegen erscheinen sonst als Wand). |
| `tb3_slam/map_merger.py` | Node: `/<ns>/map` → `/map` (2 Hz, TRANSIENT_LOCAL); Roboterposen aus `/<ns>/tf[_static]` für die Footprint-Freigabe. |
| `tb3_slam/frontier_utils.py` | Geteilte Frontier-Pipeline: Zellklassen, Frontier-Erkennung, Gruppierung, Phantom-Filter, Zielwahl (Mittelwert → nächste freie Zelle → nach außen schieben → sonst erste erreichbare Zelle auf dem Dijkstra-Pfad → Aussichtspunkt, wenn < 15 Zellen vom Roboter), `grid_to_world`/`world_to_grid`. Nur `OccupancyGrid` als ROS-Typ. |
| `tb3_slam/frontier_viz.py` | RViz-MarkerArray für Frontiers, Centroids und Roboterziele. |
| `tb3_slam/information_gain.py` | Verdeckungsbewusster Informationsgewinn (`visible_information`, `InformationField`); von beiden Explorern genutzt. Ohne ROS. |
| `tb3_slam/path_cost.py` | Schlüssel und Länge für den Nav2-Pfadkosten-Cache. Ohne ROS. |
| `tb3_slam/multi_robot_explorer.py` | Basisnode `MultiRobotExplorerNode` + Typen (`Goal`, `RobotState`, `FrontierCandidate`, `IdleTracker`). Besitzt alle ROS-I/O: gemergte `/map`, TF-Buffer aus `/<ns>/tf[_static]`, Zielversand im `<ns>/map`-Frame, Zielüberwachung (Timeout 45 s, kein Fortschritt 8 s → Recovery: Costmaps leeren → 0,3 m zurück → 90° drehen), Pfadkosten-Cache, Marker auf `/frontiers_viz`, `exhausted`-Erkennung. Hilfen für Subklassen: `candidates_for`, `path_cost`; Strategie über `select_goals` (oder eigenes `_decide`). |
| `tb3_slam/coverage_monitor.py` | `timeout_for_world_size`, erwartete Zellen, Coverage. |
| `tb3_slam/episode_manager.py` | Beendet die Episode auf der gemergten `/map` (Coverage, stuck, Timeout; erste 15 s ignoriert); schreibt `/tmp/slam_episode_done` (überschreibt keine vorhandene Datei) und jede Sekunde `….sim_s`; publiziert `/episode/coverage`. |
| `tb3_slam/cmd_vel_relay.py` | `/<ns>/cmd_vel_smoothed` (Twist) → `/<ns>/cmd_vel` (TwistStamped für die gz-Bridge). |
| `tb3_slam/tf_relay.py` | Nur für RViz: spiegelt `/<ns>/tf` auf globales `/tf`. Der Explorer braucht es nicht. |
| `tb3_slam/nav2_gate.py` | Blockiert auf `/<ns>/navigate_to_pose`; Exit 0 bei aktivem Nav2, 1 bei Timeout. Taktgeber für Nav2-Reihenfolge und Explorer-Start. |
| `test/` | Unit-Tests ohne laufende Sim (Generator, Metadaten, Merging, Frontier-Pipeline inkl. Äquivalenz zur Referenzimplementierung, Namespacing, Launch-Verdrahtung, Sim-Speed, Info-Gain, Footprints). |

### `src/tb3_exploration`

| Datei | Inhalt |
|---|---|
| `tb3_exploration/multi_robot_explorer.py` | `ClassicExplorer`: überschreibt `_decide` – prüft Kandidaten pro Roboter seriell per Nav2-`ComputePathToPose`, weist greedy zu, merkt sich gescheiterte Ziele. |
| `tb3_exploration/cost_utility.py` | Entscheidungskern (stdlib): `Bid`, `greedy_assignment` mit marginalem Gewinn, normierten Pfadkosten und Rivalen-Abschlag. |
| `tb3_exploration/frontier_explorer.py` | Alter Ein-Roboter-Node (nicht gelauncht). |
| `test/` | Nutzen/Zuweisung, Explorer-Grenzfälle gegen Nav2-Mocks, Zielüberwachung des alten `FrontierExplorer`. |

### `src/tb3_reinforced_exploration`

| Datei | Inhalt |
|---|---|
| `tb3_reinforced_exploration/multi_robot_explorer.py` | `RLExplorer`: überschreibt `select_goals` (dezentrale Runde mit Intents), zeichnet gemeinsame Ereignisse auf, trainiert online, speichert am Episodenende. |
| `tb3_reinforced_exploration/rl_core.py` | ROS-freier Kern: lokale Beobachtung (5 Features) und globaler Zustand (4), Intent-Filter, `decentralised_round`, Reward, ε-Decay pro Episode, Replay, `QMixLearner` (`agent.pt`, `mixer.pt`, `joint_replay.pkl`, `qmix_meta.pkl`), `resolve_persist_dir`. Torch lazy. Enthält auch den alten DQN-Teil für `rl_agent`. |
| `tb3_reinforced_exploration/rl_agent.py` | Alter Ein-Roboter-DQN-Node (nicht gelauncht). |
| `test/test_rl_core.py` | Tests des Lernkerns. |

### Top-Level

| Datei | Inhalt |
|---|---|
| `training_runner.py` | Episoden-Loop: Welt erzeugen/wiederverwenden, Launch mit `SLAM_WORLD_FILE`/`SLAM_SIM_SPEED`, auf `/tmp/slam_episode_done` warten, Prozesse beenden, CSV schreiben. |
| `tools/check_world_connectivity.py` | Offline (numpy): Konnektivität und Spawn-Erreichbarkeit für beide Größen. |
| `tools/profile_frontiers.py` | Benchmark der Frontier-Pipeline (System-Python, sourced); überschreibt `/tmp/slam_world_meta.json`. |
| `tools/dump_map.py` | Speichert alle 10 s die gemergte `/map` + Roboterzellen als `.npz` (laufende Sim, sourced). |
| `tools/analyze_map_dump.py` | Offline (numpy + pillow): Dump durch die Frontier-Pipeline, PNG-Rendering. |
| `Dockerfile`, `compose.yaml`, `docker/` | Headless-Image für Training; siehe `docker/README.md`. |

### Artefakte

- `/tmp/slam_world_meta.json` — `expected_traversable_cells`, `spawns`, `world_size`; vom Generator geschrieben, von `coverage_monitor` gelesen. Bei `--world` wird das `<world>.meta.json`-Sidecar hierher kopiert.
- `/tmp/slam_episode_done` — Endgrund; von `episode_manager` oder Explorer (`exhausted`) geschrieben, vom Runner gelesen.
- `/tmp/slam_episode_done.sim_s` — Sim-Sekunden seit Start; vom Runner für `elapsed_s` gelesen.
- `/tmp/slam_world_speed.sdf` — Weltkopie mit angepasstem Physik-Timing (nur bei `SLAM_SIM_SPEED` ≠ 1).
- `/tmp/slam_map_dump/map_NNN.npz` — Ausgabe von `tools/dump_map.py`.
- `results/<dd-mm-yyyy>-<mode>-<runid>.csv` — `;`-getrennt: `timestamp, episode, mode, reason, elapsed_s` (Sim-s)`, coverage, world, world_size, wall_s, sim_speed`.
- `worlds/`, `results/`, `rl_data/` sind gitignored.
