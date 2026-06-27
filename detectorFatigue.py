"""
Driver Fatigue AI - Prototype
detector.py : Détection du visage et des points du visage (landmarks)

NOTE IMPORTANTE :
Depuis MediaPipe 0.10.31, l'ancienne API "mp.solutions.face_mesh" a été
SUPPRIMÉE par Google. On utilise donc la nouvelle API officielle
"MediaPipe Tasks" (mediapipe.tasks.python.vision.FaceLandmarker),
qui est la version actuellement maintenue.
"""

import os
import urllib.request

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# URL officielle Google du modèle Face Landmarker (téléchargé une seule fois)
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)
MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "assets", "face_landmarker.task"
)


def ensure_model_downloaded():
    """Télécharge le modèle .task s'il n'existe pas encore localement."""
    if os.path.exists(MODEL_PATH) and os.path.getsize(MODEL_PATH) > 0:
        return
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    print(f"⬇️  Téléchargement du modèle Face Landmarker vers {MODEL_PATH} ...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("✅ Modèle téléchargé.")


class FaceDetector:
    """
    Encapsule MediaPipe FaceLandmarker (nouvelle API Tasks) pour détecter
    un visage et ses 478 points (landmarks), incluant les iris.
    """

    def __init__(self, max_faces=1, min_detection_confidence=0.5,
                 min_tracking_confidence=0.5):
        ensure_model_downloaded()

        base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
        options = mp_vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.VIDEO,
            num_faces=max_faces,
            min_face_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        self.landmarker = mp_vision.FaceLandmarker.create_from_options(options)
        self._frame_timestamp_ms = 0

    def process(self, frame_bgr):
        """
        Traite une frame OpenCV (BGR) et retourne le FaceLandmarkerResult.
        """
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        # En mode VIDEO, MediaPipe exige un timestamp croissant (en ms)
        self._frame_timestamp_ms += 33  # approx. 30 fps
        result = self.landmarker.detect_for_video(mp_image, self._frame_timestamp_ms)
        return result

    def get_landmarks_px(self, result, frame_shape):
        """
        Convertit les landmarks normalisés (0-1) en coordonnées pixels (x, y).
        Retourne None si aucun visage détecté, sinon une liste de (x, y) pour le 1er visage.
        """
        if not result.face_landmarks:
            return None

        h, w = frame_shape[:2]
        face_landmarks = result.face_landmarks[0]
        points = [(int(lm.x * w), int(lm.y * h)) for lm in face_landmarks]
        return points

    def draw_face_mesh(self, frame, result):
        """
        Dessine simplement les points (landmarks) sur la frame (pour debug visuel).
        """
        if result.face_landmarks:
            h, w = frame.shape[:2]
            for face_landmarks in result.face_landmarks:
                for lm in face_landmarks:
                    x, y = int(lm.x * w), int(lm.y * h)
                    cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
        return frame

    def close(self):
        self.landmarker.close()

