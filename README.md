# Driver Fatigue AI

Système de détection de fatigue en temps réel pour conducteurs, basé sur la vision par ordinateur.

## Fonctionnalités

- Détection de fermeture prolongée des yeux (EAR)
- Détection des bâillements (MAR)
- Détection de la posture de la tête (Pitch / Yaw / Roll)
- Score de fatigue global (0-100) avec 4 niveaux
- Alarme sonore automatique en cas de fatigue élevée
- PERCLOS : pourcentage de temps yeux fermés sur 1 minute

## Niveaux de fatigue

| Score  | Niveau          |
|--------|-----------------|
| 0-20   | Vigilant        |
| 20-45  | Fatigue légère  |
| 45-70  | Fatigue élevée  |
| 70-100 | Alarme          |

## Structure du projet
cat > ~/driver-fatigue-ai/README.md << 'EOF'
# Driver Fatigue AI

Système de détection de fatigue en temps réel pour conducteurs, basé sur la vision par ordinateur.

## Fonctionnalités

- Détection de fermeture prolongée des yeux (EAR)
- Détection des bâillements (MAR)
- Détection de la posture de la tête (Pitch / Yaw / Roll)
- Score de fatigue global (0-100) avec 4 niveaux
- Alarme sonore automatique en cas de fatigue élevée
- PERCLOS : pourcentage de temps yeux fermés sur 1 minute

## Niveaux de fatigue

| Score  | Niveau          |
|--------|-----------------|
| 0-20   | Vigilant        |
| 20-45  | Fatigue légère  |
| 45-70  | Fatigue élevée  |
| 70-100 | Alarme          |

## Structure du projet
driver-fatigue-ai/

├── core/

│   ├── detector.py

│   ├── alarm.py

│   └── scoring.py

├── utils/

│   └── geometry.py

├── experiences/

│   ├── phase2_camera/

│   ├── phase3_yeux/

│   ├── phase4_clignement/

│   ├── phase5_baillement/

│   ├── phase6_fatigue/

│   └── phase7_alarme/

├── assets/

├── app.py

├── Dockerfile

└── run.sh

## Lancement avec Docker

```bash
git clone https://github.com/mbedjoko/driver-fatigue-ai.git
cd driver-fatigue-ai

docker pull mbedjoko/driver-fatigue-ai:latest

xhost +local:docker
docker run --rm \
  --device=/dev/video0 \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  --device /dev/snd \
  -v /run/user/$(id -u)/pulse:/run/user/$(id -u)/pulse \
  -e PULSE_SERVER=unix:/run/user/$(id -u)/pulse/native \
  --device /dev/dri \
  -e LIBGL_ALWAYS_SOFTWARE=1 \
  mbedjoko/driver-fatigue-ai:latest
```

## Lancement sans Docker

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python app.py
```

## Technologies

- Python 3.11
- OpenCV
- MediaPipe (Face Landmarker)
- pygame
- Docker

## Auteur

Mbedjoko - https://github.com/mbedjoko
