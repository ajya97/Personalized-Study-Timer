# Personalized Study Timer — Local MVP

A local, privacy-first study timer that tracks focus and distraction using webcam, keyboard, and mouse activity, with optional machine learning classification. Designed as a single-file MVP using Streamlit.

---

## Features

- **Webcam-based feature extraction** using [Mediapipe FaceMesh]:
  - Eye Aspect Ratio (EAR) to detect blinks
  - Face presence tracking
- **Keyboard & mouse activity capture** (via `pynput`)  
- **Sliding window aggregation** of input and webcam features  
- **Focus / distraction detection**:
  - Heuristic rules by default
  - Optional RandomForest classifier trained on labeled windows
- **Pomodoro timer integration**  
- **Cross-platform desktop notifications** using `plyer`  
- **Local-only**: no raw frames or keystrokes are saved; only aggregated features persist.

---

## Installation

1. Clone the repository:

```bash
git clone https://github.com/your-username/personalized-study-timer.git
cd personalized-study-timer
