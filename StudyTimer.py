#!/usr/bin/env python3
"""
study_timer.py - Single-file MVP for a local Personalized Study Timer

Features:
- Webcam feature extraction (Mediapipe FaceMesh -> EAR, face presence)
- Keyboard & mouse capture (pynput)
- Windowed aggregation (configurable window length & step)
- Heuristic "Focused"/"Distracted" classifier by default
- Option to collect labeled windows and train a RandomForest locally
- Streamlit UI: live status, timer, charts, label buttons, train button
- Local-only: no raw frames or raw keystrokes are saved. Only aggregated features are persisted.

Setup:
pip install streamlit opencv-python mediapipe pynput scikit-learn joblib pandas plyer

Run:
streamlit run study_timer.py

Notes:
- On some OSes, pynput may require accessibility permissions for global keyboard/mouse capture.
- If you don't want global input capture, stop the listeners by toggling the checkbox in UI.
"""

import threading
import time
import collections
import math
import os
import csv
import queue
import json
from datetime import datetime
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import joblib

from pynput import keyboard, mouse
import streamlit as st

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
# Configuration / constants
# -----------------------------
WINDOW_SECONDS = 5  # length of aggregation window
STEP_SECONDS = 1    # sliding step
FPS_TARGET = 10     # approximate frame capture fps
EAR_BLINK_THRESHOLD = 0.18  # eyeblink threshold (tune per-user)
MODEL_PATH = "rf_focus_detector.joblib"
LABEL_CSV = "labeled_windows.csv"

# Mediapipe indices for eyes (from Face Mesh)
LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]

# -----------------------------
# Utility / feature functions
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
# Webcam capture thread
# -----------------------------
class WebcamCapture(threading.Thread):
    def __init__(self, frame_queue, stop_event, fps=FPS_TARGET):
        super().__init__(daemon=True)
        self.frame_queue = frame_queue
        self.stop_event = stop_event
        self.fps = fps
        self.mp_face = mp.solutions.face_mesh
        self.face_mesh = self.mp_face.FaceMesh(static_image_mode=False, max_num_faces=1,
                                               refine_landmarks=True, min_detection_confidence=0.5)
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
            # push (frame, ts) into queue but don't store many frames
            try:
                self.frame_queue.put_nowait((frame.copy(), ts))
            except queue.Full:
                # drop oldest frame to maintain throughput
                try:
                    _ = self.frame_queue.get_nowait()
                    self.frame_queue.put_nowait((frame.copy(), ts))
                except Exception:
                    pass
            time.sleep(wait)
        self.cap.release()

# -----------------------------
# Activity tracker for keyboard/mouse
# -----------------------------
class ActivityTracker:
    def __init__(self):
        self.lock = threading.Lock()
        self.keystrokes = []
        self.mouse_moves = []
        self.clicks = []
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
        self._k_listener = keyboard.Listener(on_press=lambda k: self._on_key(k))
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
            m = [t for t, x, y in self.mouse_moves if t >= cutoff]
            idle = time.time() - self.last_event_time
        features = {
            "keystrokes": len(k),
            "clicks": len(c),
            "mouse_moves": len(m),
            "idle_seconds": idle
        }
        return features

# -----------------------------
# Window aggregator & feature extractor
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
        self.mp_face = mp.solutions.face_mesh
        self.face_mesh = self.mp_face.FaceMesh(static_image_mode=False, max_num_faces=1,
                                               refine_landmarks=True, min_detection_confidence=0.5)

    def extract_window_features(self, frames):
        # frames: list of (image, ts)
        ear_vals = []
        face_presence = 0
        yaw_vals = []
        blink_count_est = 0
        prev_ear = None
        for img, ts in frames:
            h, w = img.shape[:2]
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb)
            if results.multi_face_landmarks:
                face_presence += 1
                lm = results.multi_face_landmarks[0].landmark
                ear_l = eye_aspect_ratio(lm, LEFT_EYE_IDX, w, h)
                ear_r = eye_aspect_ratio(lm, RIGHT_EYE_IDX, w, h)
                ear = (ear_l + ear_r) / 2.0
                ear_vals.append(ear)
                # blink estimation simple: count ear drops below threshold after being above
                if prev_ear is not None:
                    if prev_ear > EAR_BLINK_THRESHOLD and ear <= EAR_BLINK_THRESHOLD:
                        blink_count_est += 1
                prev_ear = ear
                # placeholder yaw: not implemented robustly here (0.0)
                yaw_vals.append(0.0)
            else:
                ear_vals.append(0.0)
        feats = {
            "ear_mean": float(np.mean(ear_vals)) if ear_vals else 0.0,
            "ear_std": float(np.std(ear_vals)) if ear_vals else 0.0,
            "face_presence": float(face_presence / max(1, len(frames))),
            "yaw_mean": float(np.mean(yaw_vals)) if yaw_vals else 0.0,
            "blink_estimate": float(blink_count_est)
        }
        # merge with keyboard/mouse
        input_feats = self.activity_tracker.sample_window(self.window_seconds)
        feats.update(input_feats)
        feats["ts_end"] = time.time()
        feats["ts_start"] = feats["ts_end"] - self.window_seconds
        return feats

    def run(self):
        # Keep frames buffer of recent frames (timestamped)
        while not self.stop_event.is_set():
            # drain incoming frames
            try:
                while True:
                    item = self.frame_queue.get_nowait()
                    self.buffer.append(item)
                    # prune old frames
                    cutoff = time.time() - self.window_seconds - 1.0
                    while self.buffer and self.buffer[0][1] < cutoff:
                        self.buffer.popleft()
            except queue.Empty:
                pass

            now = time.time()
            if now - self.last_emit >= self.step_seconds:
                # create frames list for window
                cutoff = now - self.window_seconds
                frames = [f for f in list(self.buffer) if f[1] >= cutoff]
                # if we have few frames, you may still process — heuristics will handle missing face
                feats = self.extract_window_features(frames)
                try:
                    self.output_queue.put_nowait(feats)
                except queue.Full:
                    pass
                self.last_emit = now
            time.sleep(0.05)

# -----------------------------
# Classifier wrapper
# -----------------------------
class FocusClassifier:
    def __init__(self, model_path=MODEL_PATH):
        self.model_path = model_path
        self.model = None
        self.feature_order = ["ear_mean", "ear_std", "face_presence", "blink_estimate",
                              "keystrokes", "mouse_moves", "clicks", "idle_seconds"]
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
        # If model exists, use it; else use heuristic rules
        X = np.array([feats.get(k, 0.0) for k in self.feature_order]).reshape(1, -1)
        if self.model is not None:
            try:
                p = self.model.predict_proba(X)[0][1]  # probability of "Distracted"
                return float(p)
            except Exception:
                pass
        # Heuristic fallback:
        # Distracted if face missing most of window OR idle long + low input OR ear mean very low
        distracted_score = 0.0
        # face presence
        fp = feats.get("face_presence", 0.0)
        if fp < 0.5:
            distracted_score += 0.6 * (1 - fp)
        # EAR
        ear = feats.get("ear_mean", 1.0)
        if ear < 0.14:
            distracted_score += 0.3
        # idle & keystrokes
        idle = feats.get("idle_seconds", 0.0)
        ks = feats.get("keystrokes", 0)
        if idle > 10 and ks == 0:
            distracted_score += 0.4
        # clamp
        p = max(0.0, min(1.0, distracted_score))
        return float(p)

    def train_from_csv(self, csv_path=LABEL_CSV):
        if not os.path.exists(csv_path):
            raise FileNotFoundError("Label CSV not found.")
        df = pd.read_csv(csv_path)
        # ensure necessary features present
        for col in self.feature_order:
            if col not in df.columns:
                df[col] = 0.0
        X = df[self.feature_order].values
        y = df["label"].map({"Focused": 0, "Distracted": 1}).values
        X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
        clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)
        report = classification_report(y_test, preds, target_names=["Focused", "Distracted"])
        self.model = clf
        self.save_model()
        return report

# -----------------------------
# Persistence helpers (labels)
# -----------------------------
def append_label_row(feats, label, csv_path=LABEL_CSV):
    header = ["ts_start", "ts_end", "ear_mean", "ear_std", "face_presence", "blink_estimate",
              "keystrokes", "mouse_moves", "clicks", "idle_seconds", "label"]
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(header)
        row = [
            feats.get("ts_start", 0.0),
            feats.get("ts_end", 0.0),
            feats.get("ear_mean", 0.0),
            feats.get("ear_std", 0.0),
            feats.get("face_presence", 0.0),
            feats.get("blink_estimate", 0.0),
            feats.get("keystrokes", 0),
            feats.get("mouse_moves", 0),
            feats.get("clicks", 0),
            feats.get("idle_seconds", 0.0),
            label
        ]
        writer.writerow(row)

# -----------------------------
# Streamlit App
# -----------------------------
def main():
    st.set_page_config(page_title="Personalized Study Timer", layout="wide")
    st.title("Personalized Study Timer — Local MVP")

    # Sidebar controls
    with st.sidebar:
        st.header("Settings")
        window_seconds = st.number_input("Window length (sec)", value=WINDOW_SECONDS, min_value=2, max_value=15, step=1)
        step_seconds = st.number_input("Step (sec)", value=STEP_SECONDS, min_value=1, max_value=window_seconds, step=1)
        ear_thresh = st.slider("EAR blink threshold (for heuristics)", min_value=0.05, max_value=0.4, value=EAR_BLINK_THRESHOLD, step=0.01)
        capture_inputs = st.checkbox("Capture global keyboard & mouse", value=True)
        enable_notifications = st.checkbox("Enable desktop notifications", value=True)
        st.markdown("---")
        st.write("Model")
        if os.path.exists(MODEL_PATH):
            st.success(f"Model found at {MODEL_PATH}")
        else:
            st.info("No trained model found. Will use heuristic classifier until you train.")
        if st.button("Delete saved model"):
            try:
                os.remove(MODEL_PATH)
                st.success("Deleted model.")
            except Exception as e:
                st.error(f"Error deleting model: {e}")

    # Session state initialization
    if "initialized" not in st.session_state:
        st.session_state.initialized = True
        st.session_state.stop_event = threading.Event()
        st.session_state.frame_queue = queue.Queue(maxsize=200)
        st.session_state.feature_queue = queue.Queue(maxsize=200)
        st.session_state.activity_tracker = ActivityTracker()
        st.session_state.webcam_thread = WebcamCapture(st.session_state.frame_queue, st.session_state.stop_event, fps=FPS_TARGET)
        st.session_state.aggregator = WindowAggregator(st.session_state.frame_queue, st.session_state.activity_tracker,
                                                       st.session_state.feature_queue, st.session_state.stop_event,
                                                       window_seconds=window_seconds, step_seconds=step_seconds)
        st.session_state.classifier = FocusClassifier()
        st.session_state.recent_probs = collections.deque(maxlen=300)
        st.session_state.recent_feats = collections.deque(maxlen=300)
        st.session_state.timer_running = False
        st.session_state.session_end_time = None
        st.session_state.focus_seconds = 25 * 60
        st.session_state.break_seconds = 5 * 60
        st.session_state.smooth_prob = 0.0

    # Apply GUI settings to threads where possible
    st.session_state.aggregator.window_seconds = window_seconds
    st.session_state.aggregator.step_seconds = step_seconds

    # Start/Stop capture buttons
    col_control = st.columns([3, 1])[0]
    with col_control:
        start_capture = st.button("Start capture")
        stop_capture = st.button("Stop capture")
    if start_capture:
        # start listeners and threads
        st.session_state.stop_event.clear()
        if capture_inputs:
            st.session_state.activity_tracker.start()
        # start webcam thread if not alive
        if not st.session_state.webcam_thread.is_alive():
            st.session_state.webcam_thread = WebcamCapture(st.session_state.frame_queue, st.session_state.stop_event, fps=FPS_TARGET)
            st.session_state.webcam_thread.start()
        if not st.session_state.aggregator.is_alive():
            st.session_state.aggregator = WindowAggregator(st.session_state.frame_queue, st.session_state.activity_tracker,
                                                           st.session_state.feature_queue, st.session_state.stop_event,
                                                           window_seconds=window_seconds, step_seconds=step_seconds)
            st.session_state.aggregator.start()
        st.success("Capture started (webcam + inputs).")

    if stop_capture:
        st.session_state.stop_event.set()
        st.session_state.activity_tracker.stop()
        st.success("Stopped capture.")

    # Live status & controls
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

    # Labeling / training controls
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
        if st.button("Train RandomForest from labels"):
            try:
                report = st.session_state.classifier.train_from_csv(LABEL_CSV)
                st.success("Model trained & saved.")
                st.text("Evaluation on hold-out test set:")
                st.text(report)
                st.session_state.classifier.load_model()
            except Exception as e:
                st.error(f"Training failed: {e}")

    st.markdown("---")
    st.write("Recent labeled data file:", LABEL_CSV, " — exists:" , os.path.exists(LABEL_CSV))

    # Main loop: poll feature_queue and update UI
    prob_series = []
    timestamps = []
    last_notify_time = 0
    notify_cooldown = 60  # secs between notifications

    # provide a placeholder for the loop to run in Streamlit without blocking
    placeholder = st.empty()
    # run a short loop that reads a few windows to update the page (Streamlit will re-run on user interaction)
    iter_count = 0
    while iter_count < 20:
        iter_count += 1
        try:
            feats = st.session_state.feature_queue.get(timeout=0.5)
            # record recent feats
            st.session_state.recent_feats.append(feats)
            # predict
            p = st.session_state.classifier.predict_proba(feats)
            # smooth prob (EMA)
            alpha = 0.35
            st.session_state.smooth_prob = alpha * p + (1 - alpha) * st.session_state.smooth_prob
            st.session_state.recent_probs.append(st.session_state.smooth_prob)
            # UI updates
            prob_series = list(st.session_state.recent_probs)
            timestamps = list(range(len(prob_series)))
            prob_chart.line_chart(pd.DataFrame({"distracted_prob": prob_series}))
            # status text
            status_text = f"Distracted probability: {st.session_state.smooth_prob:.2f}"
            if st.session_state.smooth_prob < 0.25:
                status_box.success(f"Status: Focused — {status_text}")
            elif st.session_state.smooth_prob < 0.6:
                status_box.info(f"Status: Neutral — {status_text}")
            else:
                status_box.error(f"Status: Distracted — {status_text}")
                # send notification if enabled and cooldown passed and pomodoro is running
                if enable_notifications and PLYER_AVAILABLE and (time.time() - last_notify_time > notify_cooldown):
                    try:
                        notification.notify(title="Study Timer: Distracted detected",
                                            message="You appear distracted. Consider a short break or refocus.",
                                            timeout=5)
                        last_notify_time = time.time()
                    except Exception:
                        pass
            # show last features (small)
            last_feats_box.json({
                "ear_mean": round(feats.get("ear_mean", 0.0), 3),
                "face_presence": round(feats.get("face_presence", 0.3), 3),
                "keystrokes": int(feats.get("keystrokes", 0)),
                "mouse_moves": int(feats.get("mouse_moves", 0)),
                "idle_sec": round(feats.get("idle_seconds", 0.0), 1),
            })
            # Pomodoro logic: if running and distracted sustained beyond threshold, optionally pause or notify user
            if st.session_state.timer_running and st.session_state.session_end_time:
                remaining = int(st.session_state.session_end_time - time.time())
                if remaining <= 0:
                    # switch between focus and break
                    if st.session_state.timer_running:
                        # simple cycle: if in focus -> break; if in break -> focus
                        if "on_break" not in st.session_state or not st.session_state.on_break:
                            st.session_state.on_break = True
                            st.session_state.session_end_time = time.time() + st.session_state.break_seconds
                            st.success("Focus period ended. Break started.")
                            if enable_notifications and PLYER_AVAILABLE:
                                try:
                                    notification.notify(title="Focus complete", message="Time for a break!", timeout=5)
                                except Exception:
                                    pass
                        else:
                            st.session_state.on_break = False
                            st.session_state.session_end_time = time.time() + st.session_state.focus_seconds
                            st.success("Break ended. Focus started.")
                            if enable_notifications and PLYER_AVAILABLE:
                                try:
                                    notification.notify(title="Break over", message="Back to focus!", timeout=5)
                                except Exception:
                                    pass
                else:
                    mins = remaining // 60
                    secs = remaining % 60
                    st.info(f"Time remaining: {mins:02d}:{secs:02d}")
            # small sleep to avoid hot-looping
            time.sleep(0.05)
        except queue.Empty:
            # no new window - show whatever we have
            if len(st.session_state.recent_probs) > 0:
                prob_chart.line_chart(pd.DataFrame({"distracted_prob": list(st.session_state.recent_probs)}))
            else:
                prob_chart.text("Waiting for data...")
            time.sleep(0.2)
        # re-evaluate whether capture is running; if not, break loop to avoid busy wait
        if st.session_state.stop_event.is_set():
            break

    placeholder.empty()
    st.write("App idle. Press 'Start capture' to begin or interact with controls. Refresh page to restart components.")

if __name__ == "__main__":
    main()
