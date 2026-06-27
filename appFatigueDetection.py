"""
Driver Fatigue AI - Prototype
Phase 7 : Déclencher une alarme sonore (pygame)
"""

import time

import cv2
from detector import FaceDetector
from utils import (
    LEFT_EYE_INDICES, RIGHT_EYE_INDICES, MOUTH_INDICES,
    average_ear, mouth_aspect_ratio,
    EyeStateTracker, YawnTracker,
)
from fatigue import FatigueScorer
from alarm import AlarmPlayer

# Seuil de score à partir duquel l'alarme sonore se déclenche.
# (Au-delà de ce seuil, le niveau "Alarme" est atteint - voir fatigue.py)
ALARM_SCORE_THRESHOLD = 70


def score_color(score):
    """Retourne une couleur BGR selon le niveau de score (vert -> rouge)."""
    if score < 20:
        return (0, 200, 0)        # vert
    elif score < 45:
        return (0, 200, 200)      # jaune
    elif score < 70:
        return (0, 140, 255)      # orange
    else:
        return (0, 0, 255)        # rouge


def draw_fatigue_bar(frame, score, x, y, width=300, height=25):
    """Dessine une barre de progression représentant le score de fatigue."""
    cv2.rectangle(frame, (x, y), (x + width, y + height), (60, 60, 60), -1)
    filled_width = int(width * (score / 100.0))
    cv2.rectangle(frame, (x, y), (x + filled_width, y + height), score_color(score), -1)
    cv2.rectangle(frame, (x, y), (x + width, y + height), (255, 255, 255), 2)
    cv2.putText(frame, f"{score:.0f}/100", (x + width + 10, y + height - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)


def main():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌ Impossible d'ouvrir la webcam. Vérifie /dev/video0 (ls /dev/video*).")
        return

    detector = FaceDetector(max_faces=1)
    eye_tracker = EyeStateTracker()
    yawn_tracker = YawnTracker()
    fatigue_scorer = FatigueScorer()
    alarm = AlarmPlayer()

    print("✅ Webcam ouverte. Appuie sur 'q' pour quitter.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ Impossible de lire une frame depuis la webcam.")
            break

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        now = time.time()

        result = detector.process(frame)
        landmarks = detector.get_landmarks_px(result, frame.shape)

        ear = average_ear(landmarks) if landmarks is not None else None
        mar = mouth_aspect_ratio(landmarks) if landmarks is not None else None

        eye_state = eye_tracker.update(ear, now)
        yawn_state = yawn_tracker.update(mar, now)
        fatigue = fatigue_scorer.compute(eye_state, yawn_state, current_time=now)

        # Déclenchement de l'alarme : score élevé OU alerte critique immédiate
        critical_alert = eye_state["is_drowsy_alert"] or yawn_state["is_yawning"]
        should_alarm = (fatigue["score"] >= ALARM_SCORE_THRESHOLD) or critical_alert
        alarm.set_active(should_alarm)

        if landmarks is not None:
            eye_color = (0, 0, 255) if eye_state["eyes_closed"] else (0, 255, 255)
            for idx in LEFT_EYE_INDICES + RIGHT_EYE_INDICES:
                x, y = landmarks[idx]
                cv2.circle(frame, (x, y), 2, eye_color, -1)

            mouth_color = (0, 0, 255) if yawn_state["is_yawning"] else (255, 0, 255)
            for idx in MOUTH_INDICES:
                x, y = landmarks[idx]
                cv2.circle(frame, (x, y), 2, mouth_color, -1)

            status_text = "Visage detecte"
            status_color = (0, 255, 0)
        else:
            status_text = "Aucun visage detecte"
            status_color = (0, 0, 255)

        ear_text = f"EAR: {ear:.3f}" if ear is not None else "EAR: N/A"
        mar_text = f"MAR: {mar:.3f}" if mar is not None else "MAR: N/A"
        blink_text = f"Clignements: {eye_state['blink_count']} ({fatigue['details']['blinks_per_minute']}/min)"
        yawn_count_text = f"Baillements: {yawn_state['yawn_count']} ({fatigue['details']['yawns_per_minute']}/min)"

        lines = [
            (status_text, status_color),
            (ear_text, (255, 255, 255)),
            (mar_text, (255, 255, 255)),
            (blink_text, (255, 255, 255)),
            (yawn_count_text, (255, 255, 255)),
        ]
        y0 = 30
        for text, color in lines:
            cv2.putText(frame, text, (20, y0),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            y0 += 26

        cv2.putText(frame, f"FATIGUE: {fatigue['level']}", (20, y0 + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, score_color(fatigue["score"]), 2)
        draw_fatigue_bar(frame, fatigue["score"], 20, y0 + 25)

        if should_alarm:
            cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 0, 255), 12)
            cv2.putText(frame, "!!! ALARME FATIGUE !!!", (w // 2 - 220, h - 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 3)

        cv2.imshow("Driver Fatigue AI - Phase 7", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    alarm.close()
    detector.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
