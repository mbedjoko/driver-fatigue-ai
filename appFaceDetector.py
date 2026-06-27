"""
Driver Fatigue AI - Prototype
Phase 2 : Détecter le visage avec MediaPipe Face Mesh
"""

import cv2
from detector import FaceDetector


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

        results = detector.process(frame)
        landmarks = detector.get_landmarks_px(results, frame.shape)

        if landmarks is not None:
            frame = detector.draw_face_mesh(frame, results)
            status_text = f"Visage detecte ({len(landmarks)} points)"
            color = (0, 255, 0)
        else:
            status_text = "Aucun visage detecte"
            color = (0, 0, 255)

        cv2.putText(frame, status_text, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        cv2.imshow("Driver Fatigue AI - Phase 2", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    detector.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
