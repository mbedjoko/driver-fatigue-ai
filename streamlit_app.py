"""
Driver Fatigue AI - Interface web TEMPS RÉEL (streamlit-webrtc)
streamlit_app.py : flux vidéo continu dans le navigateur (20-30 FPS),
avec alarme sonore pygame jouée localement.

Adapté à la structure du projet :
  core/detector.py   -> FaceDetector
  core/scoring.py    -> FatigueScorer
  core/alarm.py      -> AlarmPlayer
  utils/ (package)   -> EyeStateTracker, YawnTracker, HeadPoseTracker,
                        PerclosTracker, average_ear, mouth_aspect_ratio,
                        rotation_matrix_to_euler_angles, indices des yeux/bouche

ARCHITECTURE IMPORTANTE À COMPRENDRE :
streamlit-webrtc traite chaque frame vidéo dans une méthode recv() qui
tourne sur un thread séparé (la boucle asyncio d'aiortc), PAS sur le
thread principal de Streamlit. Cela impose deux règles strictes :

  1. On ne peut JAMAIS appeler de fonction st.* (st.metric, st.write, etc.)
     depuis recv() - ça plante ou ne fait rien.
  2. L'état doit être partagé de façon thread-safe entre recv() (thread
     webrtc) et le thread principal (qui affiche l'UI) - on utilise un
     threading.Lock pour ça.

Le thread principal relit périodiquement (via une boucle avec st.empty()
et de courtes pauses) le dernier état calculé par recv(), et met à jour
l'affichage en conséquence.

L'alarme pygame, elle, est jouée DIRECTEMENT depuis recv() - ce n'est pas
un appel Streamlit, donc c'est autorisé, et le son sort bien des
haut-parleurs de la machine qui exécute le script (ton PC, en local).
"""

import threading
import time
from collections import deque

import av
import cv2
import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode

from core.detector import FaceDetector
from core.scoring import FatigueScorer
from core.alarm import AlarmPlayer
from utils import (
    LEFT_EYE_INDICES, RIGHT_EYE_INDICES, MOUTH_INDICES,
    average_ear, mouth_aspect_ratio, rotation_matrix_to_euler_angles,
    EyeStateTracker, YawnTracker, HeadPoseTracker, PerclosTracker,
)

ALARM_SCORE_THRESHOLD = 70
HISTORY_LENGTH = 60  # ~60 derniers points affichés sur le graphique

st.set_page_config(page_title="Driver Fatigue AI", page_icon="🚗", layout="wide")

st.title("🚗 Driver Fatigue AI")
st.caption("Détection de fatigue en temps réel par vision par ordinateur — clique sur **START** pour démarrer la webcam.")

if "score_history" not in st.session_state:
    st.session_state.score_history = deque(maxlen=HISTORY_LENGTH)


def score_color(score):
    if score < 20:
        return "#1a9d4b"       # vert
    elif score < 45:
        return "#e0a000"       # jaune/orange
    elif score < 70:
        return "#e07000"       # orange foncé
    else:
        return "#d92b2b"       # rouge


class FatigueVideoProcessor:
    """
    Traite chaque frame vidéo reçue via WebRTC. Tous les trackers et le
    détecteur sont initialisés une seule fois (dans __init__), et l'état
    calculé à chaque frame est stocké dans self.latest_state, protégé par
    un Lock pour être lu en toute sécurité depuis le thread principal.
    """

    def __init__(self):
        self.detector = FaceDetector(max_faces=1)
        self.eye_tracker = EyeStateTracker()
        self.yawn_tracker = YawnTracker()
        self.head_tracker = HeadPoseTracker()
        self.perclos_tracker = PerclosTracker()
        self.fatigue_scorer = FatigueScorer()
        self.alarm = AlarmPlayer()

        self.lock = threading.Lock()
        self.latest_state = {
            "score": 0.0,
            "level": "Vigilant",
            "ear": None,
            "mar": None,
            "perclos": 0.0,
            "pitch": None,
            "yaw": None,
            "roll": None,
            "blink_count": 0,
            "yawn_count": 0,
            "should_alarm": False,
        }

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        h, w = img.shape[:2]
        now = time.time()

        result = self.detector.process(img)
        landmarks = self.detector.get_landmarks_px(result, img.shape)
        transform_matrix = self.detector.get_transformation_matrix(result)

        ear = average_ear(landmarks) if landmarks is not None else None
        mar = mouth_aspect_ratio(landmarks) if landmarks is not None else None
        angles = rotation_matrix_to_euler_angles(transform_matrix)

        eye_state = self.eye_tracker.update(ear, now)
        yawn_state = self.yawn_tracker.update(mar, now)
        head_state = self.head_tracker.update(angles, now)
        perclos_state = self.perclos_tracker.update(ear, now)
        fatigue = self.fatigue_scorer.compute(
            eye_state, yawn_state, head_state, perclos_state, current_time=now
        )

        critical_alert = (
            eye_state["is_drowsy_alert"]
            or yawn_state["is_yawning"]
            or head_state["is_pose_alert"]
            or perclos_state["is_perclos_alert"]
        )
        should_alarm = (fatigue["score"] >= ALARM_SCORE_THRESHOLD) or critical_alert

        # L'alarme pygame est jouée ici, directement depuis ce thread.
        # Ce n'est PAS un appel Streamlit, donc c'est sans danger.
        self.alarm.set_active(should_alarm)

        # --- Dessin sur l'image (overlay visuel) ---
        if landmarks is not None:
            eye_color = (0, 0, 255) if eye_state["eyes_closed"] else (0, 255, 255)
            for idx in LEFT_EYE_INDICES + RIGHT_EYE_INDICES:
                x, y = landmarks[idx]
                cv2.circle(img, (x, y), 2, eye_color, -1)

            mouth_color = (0, 0, 255) if yawn_state["is_yawning"] else (255, 0, 255)
            for idx in MOUTH_INDICES:
                x, y = landmarks[idx]
                cv2.circle(img, (x, y), 2, mouth_color, -1)

        if should_alarm:
            cv2.rectangle(img, (0, 0), (w - 1, h - 1), (0, 0, 255), 10)
            cv2.putText(img, "ALARME FATIGUE", (w // 2 - 180, h - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)

        # --- Mise à jour thread-safe de l'état partagé ---
        with self.lock:
            self.latest_state = {
                "score": fatigue["score"],
                "level": fatigue["level"],
                "ear": ear,
                "mar": mar,
                "perclos": perclos_state["perclos"],
                "pitch": head_state["pitch"],
                "yaw": head_state["yaw"],
                "roll": head_state["roll"],
                "blink_count": eye_state["blink_count"],
                "yawn_count": yawn_state["yawn_count"],
                "should_alarm": should_alarm,
            }

        return av.VideoFrame.from_ndarray(img, format="bgr24")

    def close(self):
        self.alarm.close()
        self.detector.close()


# ============================================================================
# Mise en page : vidéo + graphique à gauche (large), métriques à droite
# ============================================================================

col_video, col_metrics = st.columns([2, 1], gap="large")

with col_video:
    st.subheader("📹 Vidéo en direct")
    ctx = webrtc_streamer(
        key="driver-fatigue",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=FatigueVideoProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

    st.subheader("📊 Score de fatigue (60 dernières mesures)")
    chart_placeholder = st.empty()

    st.subheader("🧭 Posture de la tête")
    pose_col1, pose_col2, pose_col3 = st.columns(3)
    pitch_placeholder = pose_col1.empty()
    yaw_placeholder = pose_col2.empty()
    roll_placeholder = pose_col3.empty()

with col_metrics:
    st.subheader("📈 Métriques en direct")

    alert_placeholder = st.empty()

    with st.container(border=True):
        score_placeholder = st.empty()
        level_placeholder = st.empty()

    with st.container(border=True):
        st.markdown("**Indicateurs oculaires**")
        ear_placeholder = st.empty()
        perclos_placeholder = st.empty()
        blink_placeholder = st.empty()

    with st.container(border=True):
        st.markdown("**Bouche**")
        mar_placeholder = st.empty()
        yawn_count_placeholder = st.empty()

# ============================================================================
# Boucle de rafraîchissement : lit l'état calculé par recv() (thread webrtc)
# et met à jour l'affichage (thread principal). C'est le seul endroit où on
# a le droit d'appeler st.* avec ces données.
# ============================================================================

if ctx.state.playing and ctx.video_processor:
    while ctx.state.playing:
        with ctx.video_processor.lock:
            state = dict(ctx.video_processor.latest_state)

        st.session_state.score_history.append(state["score"])

        # --- Alerte principale, en haut de la colonne métriques ---
        if state["should_alarm"]:
            alert_placeholder.error("⚠️ **ALARME FATIGUE DÉTECTÉE**")
        else:
            alert_placeholder.success("✅ Conducteur vigilant")

        # --- Score + niveau ---
        score_placeholder.metric("Score de fatigue", f"{state['score']:.0f} / 100")
        level_placeholder.markdown(
            f"**Niveau :** <span style='color:{score_color(state['score'])}; "
            f"font-weight:600'>{state['level']}</span>",
            unsafe_allow_html=True,
        )

        # --- Indicateurs oculaires ---
        ear_placeholder.metric("EAR (ouverture des yeux)",
                                f"{state['ear']:.3f}" if state["ear"] is not None else "N/A")
        perclos_placeholder.metric("PERCLOS (60s)", f"{state['perclos'] * 100:.0f}%")
        blink_placeholder.metric("Clignements", state["blink_count"])

        # --- Bouche ---
        mar_placeholder.metric("MAR (ouverture de bouche)",
                                f"{state['mar']:.3f}" if state["mar"] is not None else "N/A")
        yawn_count_placeholder.metric("Bâillements", state["yawn_count"])

        # --- Graphique du score ---
        chart_placeholder.line_chart(
            {"Score fatigue": list(st.session_state.score_history)},
            height=220,
        )

        # --- Posture de la tête ---
        if state["pitch"] is not None:
            pitch_placeholder.metric("Pitch", f"{state['pitch']:.0f}°")
            yaw_placeholder.metric("Yaw", f"{state['yaw']:.0f}°")
            roll_placeholder.metric("Roll", f"{state['roll']:.0f}°")
        else:
            pitch_placeholder.metric("Pitch", "N/A")
            yaw_placeholder.metric("Yaw", "N/A")
            roll_placeholder.metric("Roll", "N/A")

        time.sleep(0.2)  # ~5 rafraîchissements/seconde, suffisant pour l'UI
else:
    with col_metrics:
        st.info("Clique sur **START** dans le flux vidéo pour démarrer l'analyse.")
