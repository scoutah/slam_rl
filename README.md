# SLAM RL

TurtleBot3-basiertes SLAM-Projekt mit klassischer Frontier-Exploration und Reinforcement-Learning-basierter Exploration in zufällig generierten Umgebungen.

## Pakete

| Paket | Beschreibung |
|---|---|
| `tb3_slam` | Simulationsstart, Welten-Generierung, Episodenverwaltung |
| `tb3_exploration` | Klassische Frontier-basierte Exploration |
| `tb3_reinforced_exploration` | RL-Agent (in Entwicklung) |

---

## Installation

### 1. ROS2 Jazzy

[Offizielle Anleitung](https://docs.ros.org/en/jazzy/Installation/Ubuntu-Install-Debs.html)

### 2. Gazebo

[Offizielle Anleitung](https://gazebosim.org/docs/harmonic/install_ubuntu/)

### 3. TurtleBot3 & Abhängigkeiten

```bash
sudo apt update
sudo apt install \
  ros-jazzy-turtlebot3 \
  ros-jazzy-turtlebot3-simulations \
  ros-jazzy-slam-toolbox \
  ros-jazzy-cartographer-ros \
  ros-jazzy-nav2-bringup
```

### 4. Repository klonen

```bash
cd ~/projects  # oder beliebiger Ordner

git clone git@github.com:scoutah/slam_rl.git
# oder: git clone https://github.com/scoutah/slam_rl.git

cd slam_rl
```

### 5. Umgebungsvariablen einrichten

```bash
# Fügt ROS2-Setup und TurtleBot3-Modell dauerhaft zur ~/.bashrc hinzu
grep -qxF 'source /opt/ros/jazzy/setup.bash' ~/.bashrc \
  || echo 'source /opt/ros/jazzy/setup.bash' >> ~/.bashrc
grep -qxF 'export TURTLEBOT3_MODEL=burger' ~/.bashrc \
  || echo 'export TURTLEBOT3_MODEL=burger' >> ~/.bashrc

source ~/.bashrc
```

### 6. Projekt bauen

```bash
# Immer im Root-Verzeichnis slam_rl/ ausführen — NICHT in src/ oder Unterpaketen!
colcon build
source install/setup.bash
```

> **Hinweis:** Nach jedem `colcon build` muss `source install/setup.bash` im jeweiligen Terminal erneut ausgeführt werden.

---

## Simulation starten

### Einmalige Simulation

Startet Gazebo, SLAM (Cartographer), Nav2 und wahlweise den Exploration-Agenten:

```bash
# Klassische Frontier-Exploration (Standard)
ros2 launch tb3_slam sim_slam.launch.py

# Oder explizit mit Exploration-Modus
ros2 launch tb3_slam sim_slam.launch.py exploration_mode:=classic

# RL-Agent statt Frontier-Explorer
ros2 launch tb3_slam sim_slam.launch.py exploration_mode:=rl
```

Der Exploration-Node startet automatisch mit ~12 s Verzögerung, damit Cartographer und Nav2 bereit sind.

**Simulation immer mit `Ctrl+C` beenden.**

### Manuelle Steuerung (separates Terminal)

```bash
source install/setup.bash
ros2 run turtlebot3_teleop teleop_keyboard
```

---

## Episodisches Training

Der `training_runner.py` startet automatisch mehrere Episoden nacheinander:

1. Zufällige Welt generieren
2. Simulation starten
3. Roboter erkundet bis **99 % der Karte** kartiert sind (oder Timeout)
4. Simulation beenden → nächste Episode

```bash
# 20 Episoden mit klassischer Exploration
python3 training_runner.py --mode classic --episodes 20

# 100 Episoden mit RL-Agent, ohne Gazebo-GUI (schneller)
python3 training_runner.py --mode rl --episodes 100 --headless

# Alle Optionen
python3 training_runner.py --help
```

### Parameter

| Argument | Standard | Beschreibung |
|---|---|---|
| `--mode` | `classic` | Exploration-Modus: `classic` oder `rl` |
| `--episodes` | `10` | Anzahl Episoden |
| `--timeout` | `600` | Max. Episodendauer in Sekunden |
| `--threshold` | `0.99` | Coverage-Schwellwert für Episodenende |
| `--headless` | `false` | Gazebo-GUI deaktivieren |
| `--world` | – | Pfad zu einer bestehenden `.sdf`-Welt (überspringt die Generierung) |
| `--randomize-worlds` | `false` | Pro Episode eine neue Zufallswelt generieren |
| `--save-worlds` | `false` | Generierte Welten in `worlds/` ablegen (Standard: `/tmp/`) |

### Coverage überwachen (separates Terminal)

```bash
source install/setup.bash
ros2 topic echo /episode/coverage
```

---

## Projektstruktur

```
slam_rl/
├── src/
│   ├── tb3_slam/
│   │   ├── launch/
│   │   │   └── sim_slam.launch.py       # Haupt-Launch-File
│   │   └── tb3_slam/
│   │       ├── utils.py                 # Welten-Generierung (SDF)
│   │       ├── frontier_utils.py        # Frontier-Erkennung & Gruppierung
│   │       ├── frontier_viz.py          # Frontier-Visualisierung (RViz-Marker)
│   │       └── episode_manager.py       # Coverage-Tracking & Episodensignal
│   ├── tb3_exploration/
│   │   └── tb3_exploration/
│   │       └── frontier_explorer.py     # Klassische Exploration
│   └── tb3_reinforced_exploration/
│       └── tb3_reinforced_exploration/
│           └── rl_agent.py              # RL-Agent (Stub)
├── training_runner.py                   # Episodischer Trainings-Loop
└── README.md
```

---

## Tests

```bash
colcon build
colcon test                                    # alle Pakete
colcon test --packages-select tb3_slam        # einzelnes Paket
colcon test-result --verbose
```

---

## Häufige Probleme

**`source install/setup.bash` vergessen** → ROS2 findet das Paket nicht. Immer nach `colcon build` und in jedem neuen Terminal ausführen.

**Simulation reagiert nicht auf `Ctrl+C`** → `killall gzserver gzclient` oder `pkill -9 -f gz` im Terminal, dann neu starten.

**Nav2 startet nicht** → Die Verzögerungen im Launch-File (5 s für Cartographer, 10 s für Nav2) sind manchmal auf langsamen Systemen zu kurz. In `sim_slam.launch.py` die `period`-Werte erhöhen.
