"""
Driver Fatigue AI - Prototype
Phase 3 : Détecter les yeux et calculer le EAR (Eye Aspect Ratio)
"""

import cv2
from detector import FaceDetector
from utils import LEFT_EYE_INDICES, RIGHT_EYE_INDICES, average_ear


def main():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌ Impossible d'ouvrir la webcam. Vérifie /dev/video0 (ls /dev/video*).")
        return

    detector = FaceDetector(max_faces=1)

    print("✅ Webcam ouverte. Appuie sur 'q' pour quitter.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ Impossible de lire une frame depuis la webcam.")
            break

        frame = cv2.flip(frame, 1)

        result = detector.process(frame)
        landmarks = detector.get_landmarks_px(result, frame.shape)

        if landmarks is not None:
            # Dessine uniquement les points des yeux (en jaune), plus visibles
            # que le nuage complet de 478 points.
            for idx in LEFT_EYE_INDICES + RIGHT_EYE_INDICES:
                x, y = landmarks[idx]
                cv2.circle(frame, (x, y), 2, (0, 255, 255), -1)

            ear = average_ear(landmarks)

            if ear is not None:
                ear_text = f"EAR: {ear:.3f}"
                status_text = "Visage detecte"
                color = (0, 255, 0)
            else:
                ear_text = "EAR: N/A"
                status_text = "Visage detecte (yeux non mesurables)"
                color = (0, 165, 255)
        else:
            ear_text = "EAR: N/A"
            status_text = "Aucun visage detecte"
            color = (0, 0, 255)

        cv2.putText(frame, status_text, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(frame, ear_text, (20, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        cv2.imshow("Driver Fatigue AI - Phase 3", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    detector.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
