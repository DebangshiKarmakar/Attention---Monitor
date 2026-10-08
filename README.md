# Classroom / Office Attention Monitor

A computer-vision project in Python that estimates how many people in a camera feed are **attentive**, **distracted**, or **drowsy**, using only a standard webcam, IP camera, or recorded video. It works for a single person or a whole classroom or office, and produces a live overlay, an optional CSV log, and a Streamlit dashboard.

> **Note:** This system measures *behavioral signals that correlate with attention* (eye closure, yawning, head direction). It does not measure attention itself. See [Limitations](#limitations) and [Privacy and ethics](#privacy-and-ethics).

---

## Features

- Multi-face detection and tracking, with a stable ID per person across frames
- Per-person state: `ATTENTIVE`, `DISTRACTED`, `DROWSY`, `NO_FACE`
- Live overlay with colored boxes, state labels, and a rolling attentiveness percentage
- Class-level summary counts (for example, "18 attentive, 4 drowsy, 3 distracted")
- Optional per-second CSV logging
- Streamlit dashboard for session trends
- Runs on CPU, with no special hardware required

---

## How it works

```
Video frame
   │
   ▼
MediaPipe Face Mesh  ──►  468 landmarks per face
   │
   ├──► EAR  (Eye Aspect Ratio)    → eyes closing / drowsiness
   ├──► MAR  (Mouth Aspect Ratio)  → yawning
   └──► Head pose (solvePnP)       → yaw / pitch → looking away or down
   │
   ▼
Centroid tracker  ──►  stable ID per face
   │
   ▼
Rolling state machine  ──►  ATTENTIVE / DISTRACTED / DROWSY
   │
   ▼
Overlay + CSV log + Streamlit dashboard
```

| Signal | Meaning | Default trigger |
|---|---|---|
| EAR < 0.21 | Eyes closed | 15 consecutive frames → `DROWSY` |
| MAR > 0.6 | Yawning | 15 consecutive frames → `DROWSY` |
| \|yaw\| > 25° or pitch > 20° | Looking away or down | 20 consecutive frames → `DISTRACTED` |

Requiring a signal to persist across several frames prevents a normal blink or a quick glance from being flagged.

---

## Project structure

```
attention_monitor/
├── attention_monitor.py   # Main script: video in → overlay + optional CSV out
├── utils.py               # EAR, MAR, and head-pose (solvePnP) helpers
├── attention_state.py     # Per-person rolling state machine + thresholds
├── tracker.py             # Centroid-based multi-face tracker
├── dashboard.py           # Streamlit dashboard for logged sessions
├── requirements.txt       # Python dependencies
└── README.md
```

---

## Installation

Requires **Python 3.9–3.11** (MediaPipe 0.10.14 does not support all newer Python versions).

```bash
# (optional) create a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

> `mediapipe` is pinned to `0.10.14` because newer releases drop the legacy `mp.solutions` API this project uses.

---

## Usage

**Live webcam:**
```bash
python attention_monitor.py
```

**Recorded video:**
```bash
python attention_monitor.py --source path/to/classroom.mp4
```

**Log results to CSV:**
```bash
python attention_monitor.py --log session.csv
```

**Set the maximum number of faces (match your room size):**
```bash
python attention_monitor.py --max-faces 30
```

Press `q` in the video window to quit.

### Dashboard

After (or during) a logged session:
```bash
streamlit run dashboard.py -- --log session.csv
```

The dashboard shows current counts, attention over time, a session-wide breakdown, and the average percentage attentive.

### CSV format

| Column | Description |
|---|---|
| `timestamp` | Time of the log row |
| `attentive` | People currently attentive |
| `distracted` | People currently distracted |
| `drowsy` | People currently drowsy |
| `no_face` | Tracked people with no visible face |
| `total` | Total tracked people |

---

## Tuning

All thresholds are at the top of `attention_state.py`.

| Parameter | Effect |
|---|---|
| `EAR_THRESHOLD` | Lower it if people are flagged drowsy too easily; raise it if closed eyes are missed |
| `MAR_THRESHOLD` | Raise it if talking or smiling is mistaken for yawning |
| `YAW_THRESHOLD_DEG` | Widen it for wide-angle rooms where students at the edges sit at an angle to the camera |
| `PITCH_DOWN_THRESHOLD_DEG` | Controls how far down someone can look before being flagged |
| `DROWSY_CONSEC_FRAMES` / `DISTRACTED_CONSEC_FRAMES` | Higher means fewer false positives but slower reaction |

Also adjust `max_distance` in `tracker.py` to suit your frame resolution.

---

## Datasets for training a learned model (next step)

The current version is rule-based. To replace the hand-set thresholds with a trained classifier (for example Random Forest or SVM on windowed EAR, MAR, yaw, pitch, and blink-rate features), these public datasets are good candidates. Check each dataset's license and access terms, as some require an academic-use request.

| Dataset | Best for | Notes |
|---|---|---|
| **DAiSEE** | Student engagement, boredom, confusion, frustration | Closest match to classroom attention; e-learning video clips |
| **UTA-RLDD** | Drowsiness in natural (non-driving) settings | Alert / low-vigilance / drowsy labels |
| **NTHU-DDD** | Drowsiness and yawning | Originally for driving; transfers reasonably to desk settings |
| **YawDD** | Yawning detection | Useful for validating MAR |
| **MRL Eye Dataset** | Open/closed eye classification | Large set of eye crops; good for a small CNN |

**Suggested approach:**
1. Run the existing landmark pipeline over dataset videos to extract per-frame features.
2. Aggregate features over sliding windows (for example 2–3 seconds).
3. Train a lightweight classifier on the dataset labels.
4. Replace the threshold logic in `attention_state.py` with the trained model.
5. Evaluate on footage from your own target environment, since lab-collected data often transfers imperfectly.

---

## Limitations

- **Proxy, not ground truth.** Someone can stare forward with open eyes and be completely zoned out.
- **Lighting and camera angle** strongly affect landmark accuracy. Backlit faces, low light, and extreme wide-angle shots need per-room tuning.
- **Occlusion** (masks, hands on face, hair over eyes) can cause `NO_FACE` or unreliable EAR/MAR readings.
- **Glasses, skin tone, and facial structure** can affect landmark detection. Test across a diverse set of people before relying on results.
- **Tracking limits.** The centroid tracker can swap IDs when people cross paths or leave and re-enter the frame.
- **Reading and note-taking** look like "looking down", so the pitch threshold may flag students who are actually working. Tune it for your setting.
- **Face size.** Faces far from the camera become too small for reliable landmarks.

---

## Privacy and ethics

Monitoring people's faces and behavior is sensitive. If you deploy this beyond a demo:

- Get **informed consent** from the people being monitored, and tell them what is measured and why.
- Prefer **aggregate, class-level statistics** over individual scoring or ranking.
- **Do not store raw video**. This project only logs counts.
- Do not use the output for disciplinary or performance-review decisions, since the signals are imperfect proxies.
- Follow your institution's policies and local data-protection law.

---

## Possible extensions

- Train a classifier on DAiSEE / UTA-RLDD features (see above)
- Add blink-rate and PERCLOS (percentage of eye closure over time) as features
- Face re-identification so IDs persist when someone leaves and returns
- Replace the centroid tracker with DeepSORT for crowded scenes
- Add gaze estimation for finer "looking at the board" detection
- Export session summaries as a PDF report

---

## Tech stack

Python · OpenCV · MediaPipe Face Mesh · NumPy · Pandas · Streamlit

---

## Acknowledgments

- Eye Aspect Ratio: Soukupová and Čech, *Real-Time Eye Blink Detection using Facial Landmarks* (2016)
- MediaPipe Face Mesh by Google
