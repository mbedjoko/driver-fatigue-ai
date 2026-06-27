"""
Driver Fatigue AI - Prototype
fatigue.py : Calcul du score de fatigue global (0-100)

Le score combine plusieurs signaux :
  - Fermeture prolongée des yeux (signal le plus fort -> micro-sommeil)
  - Fréquence des clignements sur une fenêtre récente (fatigue oculaire)
  - Fréquence des bâillements sur une fenêtre récente (signal fort de fatigue)

Le score est calculé sur une fenêtre glissante de temps (par défaut 60s),
pour refléter une tendance plutôt qu'un instant isolé.
"""

import time
from collections import deque


# Fenêtre de temps (secondes) sur laquelle on évalue la fatigue.
FATIGUE_WINDOW_SECONDS = 60.0

# Pondérations relatives de chaque facteur dans le score final (somme = 100).
WEIGHT_EYES_CLOSED = 40   # fermeture prolongée des yeux (le + critique)
WEIGHT_HEAD_POSE = 25     # tête qui tombe / penche (signal fort et précoce)
WEIGHT_BLINK_RATE = 10    # clignements anormalement fréquents
WEIGHT_YAWN_RATE = 25     # bâillements répétés

# Seuils "normaux" au-delà desquels on considère le taux anormal.
# (valeurs indicatives, à ajuster avec l'usage réel)
NORMAL_BLINKS_PER_MINUTE = 20     # un adulte cligne ~15-20 fois/minute normalement
HIGH_BLINKS_PER_MINUTE = 35       # au-delà, considéré comme signe de fatigue oculaire
MAX_YAWNS_PER_MINUTE_FOR_FULL_SCORE = 3  # 3+ bâillements/minute = score yawn max

# Niveaux de fatigue (seuils sur le score final 0-100)
LEVEL_THRESHOLDS = [
    (0, 20, "Vigilant"),
    (20, 45, "Fatigue legere"),
    (45, 70, "Fatigue elevee"),
    (70, 101, "Alarme"),
]


class FatigueScorer:
    """
    Calcule un score de fatigue (0-100) en combinant :
      - l'état actuel des yeux (fermés prolongés ou non)
      - le taux de clignement récent
      - le taux de bâillement récent
    """

    def __init__(self, window_seconds=FATIGUE_WINDOW_SECONDS):
        self.window_seconds = window_seconds
        self.blink_timestamps = deque()
        self.yawn_timestamps = deque()
        self._last_blink_count_seen = 0
        self._last_yawn_count_seen = 0

    def _prune_old(self, dq, current_time):
        """Retire les événements plus anciens que la fenêtre glissante."""
        while dq and (current_time - dq[0]) > self.window_seconds:
            dq.popleft()

    def _register_new_events(self, current_blink_count, current_yawn_count, current_time):
        """
        Détecte les NOUVEAUX clignements/bâillements depuis le dernier appel
        (en comparant les compteurs cumulés fournis par EyeStateTracker /
        YawnTracker) et les horodate dans nos fenêtres glissantes.
        """
        new_blinks = current_blink_count - self._last_blink_count_seen
        if new_blinks > 0:
            for _ in range(new_blinks):
                self.blink_timestamps.append(current_time)
        self._last_blink_count_seen = current_blink_count

        new_yawns = current_yawn_count - self._last_yawn_count_seen
        if new_yawns > 0:
            for _ in range(new_yawns):
                self.yawn_timestamps.append(current_time)
        self._last_yawn_count_seen = current_yawn_count

    def compute(self, eye_state, yawn_state, head_state=None, current_time=None):
        """
        Calcule le score de fatigue à partir des états retournés par
        EyeStateTracker.update(), YawnTracker.update(), et (optionnellement)
        HeadPoseTracker.update().

        :param eye_state: dict retourné par EyeStateTracker.update()
        :param yawn_state: dict retourné par YawnTracker.update()
        :param head_state: dict retourné par HeadPoseTracker.update(), ou None
                            si la posture de la tête n'est pas suivie.
        :param current_time: timestamp en secondes (par défaut time.time())
        :return: dict {score, level, details}
        """
        if current_time is None:
            current_time = time.time()

        self._register_new_events(
            eye_state["blink_count"], yawn_state["yawn_count"], current_time
        )
        self._prune_old(self.blink_timestamps, current_time)
        self._prune_old(self.yawn_timestamps, current_time)

        # --- Facteur 1 : yeux fermés prolongés ---
        # Score max (100% du poids) si alerte active, sinon proportionnel à
        # la durée de fermeture en cours (montée progressive avant l'alerte).
        if eye_state["is_drowsy_alert"]:
            eyes_factor = 1.0
        elif eye_state["eyes_closed"]:
            # Montée progressive entre 0 et la durée d'alerte
            eyes_factor = min(eye_state["closed_duration"] / 1.5, 1.0) * 0.6
        else:
            eyes_factor = 0.0

        # --- Facteur 2 : posture de la tête (pitch/yaw/roll anormaux) ---
        if head_state is None or head_state.get("pitch") is None:
            head_factor = 0.0
        elif head_state["is_pose_alert"]:
            head_factor = 1.0
        elif head_state["abnormal_duration"] > 0:
            # Montée progressive avant l'alerte (même logique que les yeux)
            head_factor = min(head_state["abnormal_duration"] / 2.0, 1.0) * 0.6
        else:
            head_factor = 0.0

        # --- Facteur 3 : taux de clignement récent ---
        window_minutes = self.window_seconds / 60.0
        blinks_per_minute = len(self.blink_timestamps) / window_minutes if window_minutes > 0 else 0
        if blinks_per_minute <= NORMAL_BLINKS_PER_MINUTE:
            blink_factor = 0.0
        else:
            blink_factor = min(
                (blinks_per_minute - NORMAL_BLINKS_PER_MINUTE)
                / (HIGH_BLINKS_PER_MINUTE - NORMAL_BLINKS_PER_MINUTE),
                1.0,
            )

        # --- Facteur 4 : taux de bâillement récent ---
        yawns_per_minute = len(self.yawn_timestamps) / window_minutes if window_minutes > 0 else 0
        yawn_factor = min(yawns_per_minute / MAX_YAWNS_PER_MINUTE_FOR_FULL_SCORE, 1.0)

        # --- Score final pondéré ---
        score = (
            eyes_factor * WEIGHT_EYES_CLOSED
            + head_factor * WEIGHT_HEAD_POSE
            + blink_factor * WEIGHT_BLINK_RATE
            + yawn_factor * WEIGHT_YAWN_RATE
        )
        score = round(min(max(score, 0), 100), 1)

        level = self._level_for_score(score)

        return {
            "score": score,
            "level": level,
            "details": {
                "eyes_factor": round(eyes_factor, 2),
                "head_factor": round(head_factor, 2),
                "blink_factor": round(blink_factor, 2),
                "yawn_factor": round(yawn_factor, 2),
                "blinks_per_minute": round(blinks_per_minute, 1),
                "yawns_per_minute": round(yawns_per_minute, 1),
            },
        }

    @staticmethod
    def _level_for_score(score):
        for low, high, label in LEVEL_THRESHOLDS:
            if low <= score < high:
                return label
        return "Inconnu"
