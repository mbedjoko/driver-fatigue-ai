import time
import cv2
import numpy as np
import streamlit as st
from collections import deque

from core.detector import FaceDetector
from core.scoring import FatigueScorer
from utils import (
    LEFT_EYE_INDICES, RIGHT_EYE_INDICES, MOUTH_INDICES,
    average_ear, mouth_aspect_ratio, rotation_matrix_to_euler_angles,
    EyeStateTracker, YawnTracker, HeadPoseTracker, PerclosTracker,
)

st.set_page_config(page_title="Driver Fatigue AI", layout="wide")
st.title("Driver Fatigue AI - Tableau de bord")
st.markdown("Detection de fatigue en temps reel par vision par ordinateur.")

if "score_history" not in st.session_state:
    st.session_state.score_history = deque(maxlen=60)
if "time_history" not in st.session_state:
    st.session_state.time_history = deque(maxlen=60)

@st.cache_resource
def load_components():
    detector = FaceDetector(max_faces=1)
    eye_tracker = EyeStateTracker()
    yawn_tracker = YawnTracker()
    head_tracker = HeadPoseTracker()
    perclos_tracker = PerclosTracker()
    fatigue_scorer = FatigueScorer()
    return detector, eye_tracker, yawn_tracker, head_tracker, perclos_tracker, fatigue_scorer

detector, eye_tracker, yawn_tracker, head_tracker, perclos_tracker, fatigue_scorer = load_components()

def score_color(score):
    if score < 20:
        return "green"
    elif score < 45:
        return "orange"
    elif score < 70:
        return "darkorange"
    else:
        return "red"

def draw_on_frame(frame, landmarks, eye_state, yawn_state):
    if landmarks is None:
        return frame
    eye_color = (0, 0, 255) if eye_state["eyes_closed"] else (0, 255, 255)
    for idx in LEFT_EYE_INDICES + RIGHT_EYE_INDICES:
        x, y = landmarks[idx]
        cv2.circle(frame, (x, y), 2, eye_color, -1)
    mouth_color = (0, 0, 255) if yawn_state["is_yawning"] else (255, 0, 255)
    for idx in MOUTH_INDICES:
        x, y = landmarks[idx]
        cv2.circle(frame, (x, y), 2, mouth_color, -1)
    return frame

col_video, col_metrics = st.columns([2, 1])

with col_video:
    st.subheader("Video en direct")
    video_placeholder = st.empty()
    st.subheader("Score de fatigue (60 dernieres secondes)")
    chart_placeholder = st.empty()
    st.subheader("Posture de la tete")
    pose_col1, pose_col2, pose_col3 = st.columns(3)
    pitch_placeholder = pose_col1.empty()
    yaw_placeholder = pose_col2.empty()
    roll_placeholder = pose_col3.empty()

with col_metrics:
    st.subheader("Metriques")
    score_placeholder = st.empty()
    level_placeholder = st.empty()
    st.divider()
    ear_placeholder = st.empty()
    mar_placeholder = st.empty()
    perclos_placeholder = st.empty()
    st.divider()
    blink_placeholder = st.empty()
    yawn_placeholder = st.empty()
    st.divider()
    alert_placeholder = st.empty()
    sound_placeholder = st.empty()

stop = st.button("Arreter la detection")

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    st.error("Impossible d'ouvrir la webcam.")
else:
    start_time = time.time()
    while not stop:
        ret, frame = cap.read()
        if not ret:
            st.warning("Impossible de lire la webcam.")
            break

        frame = cv2.flip(frame, 1)
        now = time.time()

        result = detector.process(frame)
        landmarks = detector.get_landmarks_px(result, frame.shape)
        transform_matrix = detector.get_transformation_matrix(result)

        ear = average_ear(landmarks) if landmarks is not None else None
        mar = mouth_aspect_ratio(landmarks) if landmarks is not None else None
        angles = rotation_matrix_to_euler_angles(transform_matrix)

        eye_state = eye_tracker.update(ear, now)
        yawn_state = yawn_tracker.update(mar, now)
        head_state = head_tracker.update(angles, now)
        perclos_state = perclos_tracker.update(ear, now)

        fatigue = fatigue_scorer.compute(
            eye_state, yawn_state, head_state, perclos_state, current_time=now
        )

        score = fatigue["score"]
        level = fatigue["level"]

        st.session_state.score_history.append(score)
        st.session_state.time_history.append(round(now - start_time, 1))

        frame = draw_on_frame(frame, landmarks, eye_state, yawn_state)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        critical = (
            eye_state["is_drowsy_alert"]
            or yawn_state["is_yawning"]
            or head_state["is_pose_alert"]
            or perclos_state["is_perclos_alert"]
        )

        if critical or score >= 70:
            h, w = frame_rgb.shape[:2]
            cv2.rectangle(frame_rgb, (0, 0), (w-1, h-1), (255, 0, 0), 12)

        video_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

        chart_placeholder.line_chart(
            {"Score fatigue": list(st.session_state.score_history)},
            height=200,
        )

        score_placeholder.metric("Score", f"{score:.0f} / 100")
        level_placeholder.markdown(
            f"**Niveau :** <span style='color:{score_color(score)}'>{level}</span>",
            unsafe_allow_html=True,
        )

        ear_placeholder.metric("EAR", f"{ear:.3f}" if ear else "N/A")
        mar_placeholder.metric("MAR", f"{mar:.3f}" if mar else "N/A")
        perclos_placeholder.metric("PERCLOS", f"{perclos_state['perclos']*100:.0f}%")

        blink_placeholder.metric("Clignements", eye_state["blink_count"])
        yawn_placeholder.metric("Baillement", yawn_state["yawn_count"])

        if head_state["pitch"] is not None:
            pitch_placeholder.metric("Pitch", f"{head_state['pitch']:.0f}")
            yaw_placeholder.metric("Yaw", f"{head_state['yaw']:.0f}")
            roll_placeholder.metric("Roll", f"{head_state['roll']:.0f}")

        if critical or score >= 70:
            alert_placeholder.error("ALERTE FATIGUE DETECTEE")
            st.components.v1.html("""
                <script>
                    var ctx = new AudioContext();
                    var oscillator = ctx.createOscillator();
                    var gainNode = ctx.createGain();
                    oscillator.connect(gainNode);
                    gainNode.connect(ctx.destination);
                    oscillator.type = 'square';
                    oscillator.frequency.setValueAtTime(880, ctx.currentTime);
                    gainNode.gain.setValueAtTime(0.3, ctx.currentTime);
                    oscillator.start();
                    oscillator.stop(ctx.currentTime + 0.5);
                </script>
            """, height=0)
        else:
            alert_placeholder.success("Conducteur vigilant")

        time.sleep(0.03)

    cap.release()
    st.info("Detection arretee.")
