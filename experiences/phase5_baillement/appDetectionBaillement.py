"""
Driver Fatigue AI - Prototype
Phase 5 : Détecter les bâillements (MAR - Mouth Aspect Ratio)
"""

import time

import cv2
from detector import FaceDetector
from utils import (
    LEFT_EYE_INDICES, RIGHT_EYE_INDICES, MOUTH_INDICES,
    average_ear, mouth_aspect_ratio,
    EyeStateTracker, YawnTracker,
)


def main():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌ Impossible d'ouvrir la webcam. Vérifie /dev/video0 (ls /dev/video*).")
        return

    detector = FaceDetector(max_faces=1)
    eye_tracker = EyeStateTracker()
    yawn_tracker = YawnTracker()

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
        eyes_text = "YEUX FERMES" if eye_state["eyes_closed"] else "Yeux ouverts"
        eyes_color = (0, 0, 255) if eye_state["eyes_closed"] else (0, 255, 0)
        yawn_text = "BAILLEMENT" if yawn_state["is_yawning"] else "Bouche normale"
        yawn_color = (0, 0, 255) if yawn_state["is_yawning"] else (0, 255, 0)
        blink_text = f"Clignements: {eye_state['blink_count']}"
        yawn_count_text = f"Baillements: {yawn_state['yawn_count']}"

        lines = [
            (status_text, status_color),
            (ear_text, (255, 255, 255)),
            (eyes_text, eyes_color),
            (mar_text, (255, 255, 255)),
            (yawn_text, yawn_color),
            (blink_text, (255, 255, 255)),
            (yawn_count_text, (255, 255, 255)),
        ]
        y0 = 30
        for text, color in lines:
            cv2.putText(frame, text, (20, y0),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
            y0 += 28

        if eye_state["is_drowsy_alert"] or yawn_state["is_yawning"]:
            cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 0, 255), 12)
            cv2.putText(frame, "!!! ALERTE FATIGUE !!!", (w // 2 - 220, h - 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 3)

        cv2.imshow("Driver Fatigue AI - Phase 5", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    detector.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
