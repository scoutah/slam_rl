# Workflow

## User-Prompt:

Der Roboter soll mithilfe von Reinforcement Learning lernen, wie man am besten eine Karte mapped. Dabei wird er belohnt, wenn er schnell ist und bestraft, wenn er bspw. in ein Sackgasse läuft. Er soll aber "von alleine laufen" und nicht mit User-Input.

## 1️⃣ Übersicht des Ablaufs

Ziel:

- TurtleBot3 fährt autonom durch die simulierte Umgebung
- SLAM (Cartographer) erstellt eine Karte
- RL-Agent bekommt Belohnung / Strafe:
- Belohnung: schnell neue Bereiche mappen
- Strafe: Sackgassen, Kollisionen, ineffizientes Verhalten

## 2️⃣ Komponenten, die du brauchst

| Komponente          | Zweck                                                                                          |
| ------------------- | ---------------------------------------------------------------------------------------------- |
| Simulation          | Gazebo mit TurtleBot3-Welt. Hier findet das Training statt.                                    |
| Robot Model         | TurtleBot3 (Burger) mit LIDAR + Odometry.                                                      |
| SLAM                | Cartographer, um Karte zu bauen. Auch der RL-Agent kann auf das aktuelle Map-Update zugreifen. |
| ROS2 Nodes / Topics | /scan, /odom, /tf, /map, /cmd_vel – RL-Agent liest Sensoren und steuert Roboter.               |
| RL-Agent / Policy   | Python Node (z. B. mit PyTorch / Stable-Baselines3), das Aktionen auf /cmd_vel veröffentlicht. |
| Reward Function     | Python-Funktion, die basierend auf Sensor- und Map-Daten bewertet.                             |
| Episode Management  | Start / Reset der Simulation für jede Episode, z. B. per ros2 service call /reset_simulation.  |
| Logging / Replay    | rosbag oder direkt NumPy / CSV, um Trainingsdaten zu speichern.                                |

## 3️⃣ Typischer Workflow

1. Simulation vorbereiten

- Launchfile startet:
  - Gazebo (welt + Roboter)
  - Cartographer (SLAM)
  - Optionale Logger (Sensoren / Map)

2. RL-Agent startet

- Abonniert /scan, /odom, /map
- Berechnet Zustand (state) für die Policy

3. Action-Publishing

- Agent schreibt kontinuierlich auf /cmd_vel

4. Reward berechnen

- Geschwindigkeit neuer Map-Regionen
- Kollisionen / Sackgassen → Strafen

5. Episode-Ende / Reset

- Roboter neu positionieren / Simulation zurücksetzen
- Map optional löschen oder weiter nutzen

6. Training / Policy Update

- Offline / online Update der RL-Policy

7. Wiederholung

- Schleife läuft automatisch, kein User-Input nötig

## 4️⃣ Tools und Bibliotheken

- Python RL Framework:
  - Stable-Baselines3 (mit PPO, DQN, SAC)
  - Vorteil: fertige Trainer, leicht in ROS2 Nodes integrierbar

- ROS2 Python Bindings:
  - rclpy zum Abonnieren / Publizieren

- Simulation:
  - Gazebo (ROS2 Integration)
  - Optional ros2_control für genaue Robotersteuerung

- Datenlogging / Replay:
  - rosbag2 für spätere Analyse
  - Oder direkt Numpy/CSV im Agenten

## 5️⃣ Struktur deines Projekts
slam_rl/
└── src/
  ├── tb3_slam/ # Launch Simulation
  └── tb3_rl_agent/ # RL Agent Node
    ├── agent.py # Policy + Reward
    ├── train.py # Trainingsloop / Episode Management
    └── launch/ # Optional Launchfile für RL-Training

## 6️⃣ Wichtige Hinweise

- RL-Agent darf nie direkt die Map schreiben – nur /cmd_vel publizieren
- Belohnungsfunktion sorgfältig definieren, sonst lernt der Roboter Quatsch
- Simulation Reset: wichtig für episodisches Lernen
- Data Logging: speichere Zustand, Aktion, Reward → für spätere Analyse oder Training
- Compute: RL kann viel Rechenleistung brauchen → evtl. Sandbox-Simulation mit kleinerer Map
