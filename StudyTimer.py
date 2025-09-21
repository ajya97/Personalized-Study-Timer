#!/usr/bin/env python3
"""
study_timer.py - Personalized Study Timer (Local MVP)

Features:
- Webcam feature extraction (Mediapipe FaceMesh -> EAR, face presence)
- Keyboard & mouse capture (pynput)
- Sliding window aggregation
- Heuristic + optional RandomForest classifier
- Streamlit UI: status, timer, charts, labeling, training
- Local-only: no raw frames stored

Setup:
    pip install streamlit opencv-python mediapipe pynput scikit-learn joblib pandas plyer

Run:
    streamlit run study_timer.py

Linux note:
    If you see "ImportError: libGL.so.1: cannot open shared object file",
    install OpenCV’s system dependency:
        sudo apt-get update && sudo apt-get install -y libgl1
"""

import os
import cv2
import csv
import time
import math
import json
import queue
import joblib
import threading
import collections
import numpy as np
import pandas as pd
import mediapipe as mp
import streamlit as st

from pathlib import Path
from datetime import datetime
from pynput import keyboard, mouse
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Optional cross-platform notifications
try:
    from plyer import notification
    PLYER_AVAILABLE = True
except Exception:
    PLYER_AVAILABLE = False

# -----------------------------
# Configuration
# -----------------------------
WINDOW_SECONDS = 5
STEP_SECONDS = 1
FPS_TARGET = 10
EAR_BLINK_THRESHOLD = 0.18
MODEL_PATH = "rf_focus_detector.joblib"
LABEL_CSV = "labeled_windows.csv"

LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]

# -----------------------------
# Utility
# -----------------------------
def hypot(a, b):
    return math.hypot(a, b)

def eye_aspect_ratio(landmarks, idxs, image_w, image_h):
    try:
        pts = [(int(landmarks[i].x * image_w), int(landmarks[i].y * image_h)) for i in idxs]
        A = hypot(pts[1][0]-pts[5][0], pts[1][1]-pts[5][1])
        B = hypot(pts[2][0]-pts[4][0], pts[2][1]-pts[4][1])
        C = hypot(pts[0][0]-pts[3][0], pts[0][1]-pts[3][1])
        if C == 0:
            return 0.0
        return (A + B) / (2.0 * C)
    except Exception:
        return 0.0

# -----------------------------
# Webcam Capture
# -----------------------------
class WebcamCapture(threading.Thread):
    def __init__(self, frame_queue, stop_event, fps=FPS_TARGET):
        super().__init__(daemon=True)
        self.frame_queue = frame_queue
        self.stop_event = stop_event
        self.fps = fps
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False, max_num_faces=1,
            refine_landmarks=True, min_detection_confidence=0.5
        )
        self.cap = None

    def run(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            st.warning("Unable to open webcam. Webcam features will be disabled.")
            return
        wait = 1.0 / max(1, self.fps)
        while not self.stop_event.is_set():
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(wait)
                continue
            ts = time.time()
            try:
                self.frame_queue.put_nowait((frame.copy(), ts))
            except queue.Full:
                try:
                    _ = self.frame_queue.get_nowait()
                    self.frame_queue.put_nowait((frame.copy(), ts))
                except Exception:
                    pass
            time.sleep(wait)
        self.cap.release()

# -----------------------------
# Keyboard/Mouse Tracker
# -----------------------------
class ActivityTracker:
    def __init__(self):
        self.lock = threading.Lock()
        self.keystrokes, self.mouse_moves, self.clicks = [], [], []
        self.last_event_time = time.time()
        self._k_listener = None
        self._m_listener = None
        self.listeners_running = False

    def _on_key(self, key):
        with self.lock:
            self.keystrokes.append(time.time())
            self.last_event_time = time.time()

    def _on_move(self, x, y):
        with self.lock:
            self.mouse_moves.append((time.time(), x, y))
            self.last_event_time = time.time()

    def _on_click(self, x, y, button, pressed):
        if pressed:
            with self.lock:
                self.clicks.append(time.time())
                self.last_event_time = time.time()

    def start(self):
        if self.listeners_running:
            return
        self._k_listener = keyboard.Listener(on_press=self._on_key)
        self._m_listener = mouse.Listener(on_move=self._on_move, on_click=self._on_click)
        self._k_listener.start()
        self._m_listener.start()
        self.listeners_running = True

    def stop(self):
        try:
            if self._k_listener:
                self._k_listener.stop()
            if self._m_listener:
                self._m_listener.stop()
        except Exception:
            pass
        self.listeners_running = False

    def sample_window(self, window_seconds=WINDOW_SECONDS):
        cutoff = time.time() - window_seconds
        with self.lock:
            k = [t for t in self.keystrokes if t >= cutoff]
            c = [t for t in self.clicks if t >= cutoff]
            m = [t for t, _, _ in self.mouse_moves if t >= cutoff]
            idle = time.time() - self.last_event_time
        return {
            "keystrokes": len(k),
            "clicks": len(c),
            "mouse_moves": len(m),
            "idle_seconds": idle,
        }

# -----------------------------
# Window Aggregator
# -----------------------------
class WindowAggregator(threading.Thread):
    def __init__(self, frame_queue, activity_tracker, output_queue, stop_event,
                 window_seconds=WINDOW_SECONDS, step_seconds=STEP_SECONDS):
        super().__init__(daemon=True)
        self.frame_queue = frame_queue
        self.activity_tracker = activity_tracker
        self.output_queue = output_queue
        self.stop_event = stop_event
        self.window_seconds = window_seconds
        self.step_seconds = step_seconds
        self.buffer = collections.deque()
        self.last_emit = 0
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False, max_num_faces=1,
            refine_landmarks=True, min_detection_confidence=0.5
        )

    def extract_window_features(self, frames):
        ear_vals, yaw_vals = [], []
        face_presence, blink_count_est = 0, 0
        prev_ear = None
        for img, ts in frames:
            h, w = img.shape[:2]
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb)
            if results.multi_face_landmarks:
                face_presence += 1
                lm = results.multi_face_landmarks[0].landmark
                ear = (eye_aspect_ratio(lm, LEFT_EYE_IDX, w, h) +
                       eye_aspect_ratio(lm, RIGHT_EYE_IDX, w, h)) / 2.0
                ear_vals.append(ear)
                if prev_ear is not None and prev_ear > EAR_BLINK_THRESHOLD and ear <= EAR_BLINK_THRESHOLD:
                    blink_count_est += 1
                prev_ear = ear
                yaw_vals.append(0.0)  # placeholder
            else:
                ear_vals.append(0.0)
        feats = {
            "ear_mean": float(np.mean(ear_vals)) if ear_vals else 0.0,
            "ear_std": float(np.std(ear_vals)) if ear_vals else 0.0,
            "face_presence": float(face_presence / max(1, len(frames))),
            "yaw_mean": float(np.mean(yaw_vals)) if yaw_vals else 0.0,
            "blink_estimate": float(blink_count_est),
        }
        feats.update(self.activity_tracker.sample_window(self.window_seconds))
        feats["ts_end"] = time.time()
        feats["ts_start"] = feats["ts_end"] - self.window_seconds
        return feats

    def run(self):
        while not self.stop_event.is_set():
            try:
                while True:
                    item = self.frame_queue.get_nowait()
                    self.buffer.append(item)
                    cutoff = time.time() - self.window_seconds - 1.0
                    while self.buffer and self.buffer[0][1] < cutoff:
                        self.buffer.popleft()
            except queue.Empty:
                pass

            now = time.time()
            if now - self.last_emit >= self.step_seconds:
                cutoff = now - self.window_seconds
                frames = [f for f in list(self.buffer) if f[1] >= cutoff]
                feats = self.extract_window_features(frames)
                try:
                    self.output_queue.put_nowait(feats)
                except queue.Full:
                    pass
                self.last_emit = now
            time.sleep(0.05)

# -----------------------------
# Classifier
# -----------------------------
class FocusClassifier:
    def __init__(self, model_path=MODEL_PATH):
        self.model_path = model_path
        self.model = None
        self.feature_order = [
            "ear_mean", "ear_std", "face_presence", "blink_estimate",
            "keystrokes", "mouse_moves", "clicks", "idle_seconds"
        ]
        self.load_model()

    def load_model(self):
        if os.path.exists(self.model_path):
            try:
                self.model = joblib.load(self.model_path)
            except Exception:
                self.model = None

    def save_model(self):
        if self.model is not None:
            joblib.dump(self.model, self.model_path)

    def predict_proba(self, feats):
        X = np.array([feats.get(k, 0.0) for k in self.feature_order]).reshape(1, -1)
        if self.model is not None:
            try:
                return float(self.model.predict_proba(X)[0][1])
            except Exception:
                pass
        # Heuristic fallback
        distracted_score = 0.0
        fp, ear = feats.get("face_presence", 0.0), feats.get("ear_mean", 1.0)
        idle, ks = feats.get("idle_seconds", 0.0), feats.get("keystrokes", 0)
        if fp < 0.5:
            distracted_score += 0.6 * (1 - fp)
        if ear < 0.14:
            distracted_score += 0.3
        if idle > 10 and ks == 0:
            distracted_score += 0.4
        return float(max(0.0, min(1.0, distracted_score)))

    def train_from_csv(self, csv_path=LABEL_CSV):
        if not os.path.exists(csv_path):
            raise FileNotFoundError("Label CSV not found.")
        df = pd.read_csv(csv_path)
        for col in self.feature_order:
            if col not in df.columns:
                df[col] = 0.0
        X = df[self.feature_order].values
        y = df["label"].map({"Focused": 0, "Distracted": 1}).values
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, stratify=y, test_size=0.2, random_state=42
        )
        clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
        clf.fit(X_train, y_train)
        report = classification_report(y_test, clf.predict(X_test),
                                       target_names=["Focused", "Distracted"])
        self.model = clf
        self.save_model()
        return report

# -----------------------------
# Persistence
# -----------------------------
def append_label_row(feats, label, csv_path=LABEL_CSV):
    header = ["ts_start", "ts_end", "ear_mean", "ear_std", "face_presence", "blink_estimate",
              "keystrokes", "mouse_moves", "clicks", "idle_seconds", "label"]
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(header)
        writer.writerow([
            feats.get("ts_start", 0.0), feats.get("ts_end", 0.0),
            feats.get("ear_mean", 0.0), feats.get("ear_std", 0.0),
            feats.get("face_presence", 0.0), feats.get("blink_estimate", 0.0),
            feats.get("keystrokes", 0), feats.get("mouse_moves", 0),
            feats.get("clicks", 0), feats.get("idle_seconds", 0.0),
            label
        ])

# -----------------------------
# Streamlit App
# -----------------------------
def main():
    st.set_page_config(page_title="Personalized Study Timer", layout="wide")
    st.title("Personalized Study Timer — Local MVP")

    # Sidebar
    with st.sidebar:
        st.header("Settings")
        window_seconds = st.number_input("Window length (sec)", value=WINDOW_SECONDS, min_value=2, max_value=15, step=1)
        step_seconds = st.number_input("Step (sec)", value=STEP_SECONDS, min_value=1, max_value=window_seconds, step=1)
        capture_inputs = st.checkbox("Capture global keyboard & mouse", value=True)
        enable_notifications = st.checkbox("Enable desktop notifications", value=True)
        st.markdown("---")
        if os.path.exists(MODEL_PATH):
            st.success(f"Model found at {MODEL_PATH}")
        else:
            st.info("No trained model found. Using heuristic classifier.")
        if st.button("Delete saved model"):
            try:
                os.remove(MODEL_PATH)
                st.success("Deleted model.")
            except Exception as e:
                st.error(f"Error deleting model: {e}")

    # Init session state
    if "initialized" not in st.session_state:
        st.session_state.initialized = True
        st.session_state.stop_event = threading.Event()
        st.session_state.frame_queue = queue.Queue(maxsize=200)
        st.session_state.feature_queue = queue.Queue(maxsize=200)
        st.session_state.activity_tracker = ActivityTracker()
        st.session_state.webcam_thread = WebcamCapture(st.session_state.frame_queue, st.session_state.stop_event)
        st.session_state.aggregator = WindowAggregator(
            st.session_state.frame_queue, st.session_state.activity_tracker,
            st.session_state.feature_queue, st.session_state.stop_event,
            window_seconds=window_seconds, step_seconds=step_seconds
        )
        st.session_state.classifier = FocusClassifier()
        st.session_state.recent_probs = collections.deque(maxlen=300)
        st.session_state.recent_feats = collections.deque(maxlen=300)
        st.session_state.timer_running = False
        st.session_state.session_end_time = None
        st.session_state.focus_seconds = 25 * 60
        st.session_state.break_seconds = 5 * 60
        st.session_state.smooth_prob = 0.0

    st.session_state.aggregator.window_seconds = window_seconds
    st.session_state.aggregator.step_seconds = step_seconds

    # Controls
    col_control = st.columns([3, 1])[0]
    with col_control:
        start_capture = st.button("Start capture")
        stop_capture = st.button("Stop capture")
    if start_capture:
        st.session_state.stop_event.clear()
        if capture_inputs:
            st.session_state.activity_tracker.start()
        if not st.session_state.webcam_thread.is_alive():
            st.session_state.webcam_thread = WebcamCapture(st.session_state.frame_queue, st.session_state.stop_event)
            st.session_state.webcam_thread.start()
        if not st.session_state.aggregator.is_alive():
            st.session_state.aggregator = WindowAggregator(
                st.session_state.frame_queue, st.session_state.activity_tracker,
                st.session_state.feature_queue, st.session_state.stop_event,
                window_seconds=window_seconds, step_seconds=step_seconds
            )
            st.session_state.aggregator.start()
        st.success("Capture started.")

    if stop_capture:
        st.session_state.stop_event.set()
        st.session_state.activity_tracker.stop()
        st.success("Stopped capture.")

    # UI Layout
    left, right = st.columns([2, 1])
    with left:
        st.subheader("Live distraction status")
        status_box = st.empty()
        prob_chart = st.empty()
        last_feats_box = st.empty()
    with right:
        st.subheader("Pomodoro timer")
        st.session_state.focus_seconds = st.number_input("Focus length (minutes)", min_value=1, max_value=180, value=25) * 60
        st.session_state.break_seconds = st.number_input("Break length (minutes)", min_value=1, max_value=60, value=5) * 60
        if st.button("Start Pomodoro Session"):
            st.session_state.timer_running = True
            st.session_state.session_end_time = time.time() + st.session_state.focus_seconds
            st.success("Pomodoro started.")
        if st.button("Stop Pomodoro"):
            st.session_state.timer_running = False
            st.session_state.session_end_time = None
            st.info("Pomodoro stopped.")

    # Labeling & Training
    st.markdown("---")
    st.subheader("Labeling & model")
    label_col1, label_col2, label_col3 = st.columns([1,1,2])
    with label_col1:
        if st.button("Label last window: Focused"):
            try:
                feats = st.session_state.recent_feats[-1]
                append_label_row(feats, "Focused")
                st.success("Saved label Focused.")
            except Exception:
                st.error("No recent window to label.")
    with label_col2:
        if st.button("Label last window: Distracted"):
            try:
                feats = st.session_state.recent_feats[-1]
                append_label_row(feats, "Distracted")
                st.success("Saved label Distracted.")
            except Exception:
                st.error("No recent window to label.")
    with label_col3:
        if st.button("Train model from CSV"):
            try:
                report = st.session_state.classifier.train_from_csv()
                st.code(report)
                st.success("Model trained and saved.")
            except Exception as e:
                st.error(str(e))

    # Poll feature queue
    try:
        while True:
            feats = st.session_state.feature_queue.get_nowait()
            prob = st.session_state.classifier.predict_proba(feats)
            st.session_state.smooth_prob = 0.7 * st.session_state.smooth_prob + 0.3 * prob
            st.session_state.recent_probs.append((feats["ts_end"], st.session_state.smooth_prob))
            st.session_state.recent_feats.append(feats)
    except queue.Empty:
        pass

    # Update UI
    if st.session_state.recent_probs:
        last_ts, prob = st.session_state.recent_probs[-1]
        status = "Distracted" if prob > 0.5 else "Focused"
        status_color = "red" if status == "Distracted" else "green"
        status_box.markdown(f"<h2 style='color:{status_color};'>Status: {status} ({prob:.2f})</h2>", unsafe_allow_html=True)
        df_chart = pd.DataFrame(st.session_state.recent_probs, columns=["ts", "prob"])
        df_chart["time"] = pd.to_datetime(df_chart["ts"], unit="s")
        prob_chart.line_chart(df_chart.set_index("time")["prob"])
        last_feats_box.json(st.session_state.recent_feats[-1])
        if enable_notifications and PLYER_AVAILABLE and status == "Distracted":
            notification.notify(title="Study Timer Alert", message="You seem distracted!", timeout=3)

    if st.session_state.timer_running and st.session_state.session_end_time:
        remaining = max(0, int(st.session_state.session_end_time - time.time()))
        mins, secs = divmod(remaining, 60)
        st.sidebar.markdown(f"⏳ **Time left:** {mins:02d}:{secs:02d}")
        if remaining <= 0:
            st.session_state.timer_running = False
            st.sidebar.success("Session complete! Take a break.")
            if enable_notifications and PLYER_AVAILABLE:
                notification.notify(title="Pomodoro Complete", message="Time for a break!", timeout=5)

if __name__ == "__main__":
    main()
