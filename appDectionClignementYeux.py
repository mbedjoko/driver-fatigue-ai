"""
Driver Fatigue AI - Prototype
Phase 4 : Détecter les yeux fermés prolongés (signe de fatigue)
"""

import time

import cv2
from detector import FaceDetector
from utils import LEFT_EYE_INDICES, RIGHT_EYE_INDICES, average_ear, EyeStateTracker


def main():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌ Impossible d'ouvrir la webcam. Vérifie /dev/video0 (ls /dev/video*).")
        return

    detector = FaceDetector(max_faces=1)
    eye_tracker = EyeStateTracker()

    print("✅ Webcam ouverte. Appuie sur 'q' pour quitter.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ Impossible de lire une frame depuis la webcam.")
            break

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        result = detector.process(frame)
        landmarks = detector.get_landmarks_px(result, frame.shape)

        ear = average_ear(landmarks) if landmarks is not None else None
        state = eye_tracker.update(ear, time.time())

        if landmarks is not None:
            eye_color = (0, 0, 255) if state["eyes_closed"] else (0, 255, 255)
            for idx in LEFT_EYE_INDICES + RIGHT_EYE_INDICES:
                x, y = landmarks[idx]
                cv2.circle(frame, (x, y), 2, eye_color, -1)

            status_text = "Visage detecte"
            status_color = (0, 255, 0)
        else:
            status_text = "Aucun visage detecte"
            status_color = (0, 0, 255)

        ear_text = f"EAR: {ear:.3f}" if ear is not None else "EAR: N/A"
        eyes_text = "YEUX FERMES" if state["eyes_closed"] else "Yeux ouverts"
        eyes_color = (0, 0, 255) if state["eyes_closed"] else (0, 255, 0)
        duration_text = f"Duree fermeture: {state['closed_duration']:.1f}s"
        blink_text = f"Clignements: {state['blink_count']}"

        cv2.putText(frame, status_text, (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        cv2.putText(frame, ear_text, (20, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, eyes_text, (20, 95),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, eyes_color, 2)
        cv2.putText(frame, duration_text, (20, 125),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, blink_text, (20, 155),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        if state["is_drowsy_alert"]:
            cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 0, 255), 12)
            cv2.putText(frame, "!!! ALERTE FATIGUE !!!", (w // 2 - 220, h - 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 3)

        cv2.imshow("Driver Fatigue AI - Phase 4", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    detector.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
