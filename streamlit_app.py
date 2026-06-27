import time
import cv2
import numpy as np
import streamlit as st
from collections import deque
from PIL import Image
from detector import FaceDetector
from fatigue import FatigueScorer
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
if "eye_tracker" not in st.session_state:
    st.session_state.eye_tracker = EyeStateTracker()
if "yawn_tracker" not in st.session_state:
    st.session_state.yawn_tracker = YawnTracker()
if "head_tracker" not in st.session_state:
    st.session_state.head_tracker = HeadPoseTracker()
if "perclos_tracker" not in st.session_state:
    st.session_state.perclos_tracker = PerclosTracker()
if "fatigue_scorer" not in st.session_state:
    st.session_state.fatigue_scorer = FatigueScorer()
@st.cache_resource
def load_detector():
    return FaceDetector(max_faces=1)
detector = load_detector()
def score_color(score):
    if score < 20:
        return "green"
    elif score < 45:
        return "orange"
    elif score < 70:
        return "darkorange"
    else:
        return "red"
col_video, col_metrics = st.columns([2, 1])
with col_video:
    st.subheader("Capture webcam")
    img_file = st.camera_input("Activer la camera")
    st.subheader("Score de fatigue (60 dernieres captures)")
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
    yawn_count_placeholder = st.empty()
    st.divider()
    alert_placeholder = st.empty()
if img_file is not None:
    img = Image.open(img_file)
    frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    now = time.time()
    result = detector.process(frame)
    landmarks = detector.get_landmarks_px(result, frame.shape)
    transform_matrix = detector.get_transformation_matrix(result)
    ear = average_ear(landmarks) if landmarks is not None else None
    mar = mouth_aspect_ratio(landmarks) if landmarks is not None else None
    angles = rotation_matrix_to_euler_angles(transform_matrix)
    eye_state = st.session_state.eye_tracker.update(ear, now)
    yawn_state = st.session_state.yawn_tracker.update(mar, now)
    head_state = st.session_state.head_tracker.update(angles, now)
    perclos_state = st.session_state.perclos_tracker.update(ear, now)
    fatigue = st.session_state.fatigue_scorer.compute(
        eye_state, yawn_state, head_state, perclos_state, current_time=now
    )
    score = fatigue["score"]
    level = fatigue["level"]
    st.session_state.score_history.append(score)
    if landmarks is not None:
        eye_color = (0, 0, 255) if eye_state["eyes_closed"] else (0, 255, 255)
        for idx in LEFT_EYE_INDICES + RIGHT_EYE_INDICES:
            x, y = landmarks[idx]
            cv2.circle(frame, (x, y), 2, eye_color, -1)
        mouth_color = (0, 0, 255) if yawn_state["is_yawning"] else (255, 0, 255)
        for idx in MOUTH_INDICES:
            x, y = landmarks[idx]
            cv2.circle(frame, (x, y), 2, mouth_color, -1)
    critical = (
        eye_state["is_drowsy_alert"]
        or yawn_state["is_yawning"]
        or head_state["is_pose_alert"]
        or perclos_state["is_perclos_alert"]
    )
    if critical or score >= 70:
        h, w = frame.shape[:2]
        cv2.rectangle(frame, (0, 0), (w-1, h-1), (255, 0, 0), 12)
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
    yawn_count_placeholder.metric("Baillements", yawn_state["yawn_count"])
    if head_state["pitch"] is not None:
        pitch_placeholder.metric("Pitch", f"{head_state['pitch']:.0f}")
        yaw_placeholder.metric("Yaw", f"{head_state['yaw']:.0f}")
        roll_placeholder.metric("Roll", f"{head_state['roll']:.0f}")
    if critical or score >= 70:
        alert_placeholder.error("ALERTE FATIGUE DETECTEE")
    else:
        alert_placeholder.success("Conducteur vigilant")
