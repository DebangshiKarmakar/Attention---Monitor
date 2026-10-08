""""Main entry point for the attention monitor application.

Works for both single webcam and classroom or office feeds with multiple people.
It detects every face independently.
Usage:
    python attention_monitor.py       # webcam
Press 'q' to quit.
The Face Landmarker model is downloaded to the user cache on first run.
"""

import argparse
import csv
import hashlib
import math
import os
import time
import urllib.request
from pathlib import Path
from tempfile import NamedTemporaryFile

import cv2
import mediapipe as mp

from attention_state import AttentionRegistry
from tracker import CentroidTracker
from utils import (
    LEFT_EYE,
    MOUTH,
    RIGHT_EYE,
    eye_aspect_ratio,
    get_head_pose,
    landmarks_to_xy,
    mouth_aspect_ratio,
)

STATE_COLORS = {
    "ATTENTIVE": (0, 200, 0),
    "DISTRACTED": (0, 165, 255),
    "DROWSY": (0, 0, 255),
    "NO FACE": (128, 128, 128),
}

FACE_LANDMARKER_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)
FACE_LANDMARKER_MODEL_MD5 = "b0e7274907a1644404fef66b28dd6d85"


def _default_model_path():
    cache_root = os.environ.get("LOCALAPPDATA")
    if cache_root:
        cache_dir = Path(cache_root) / "AttentionMonitor"
    else:
        cache_dir = Path.home() / ".cache" / "attention-monitor"
    return cache_dir / "face_landmarker.task"


def _ensure_model(model_path=None):
    if model_path:
        path = Path(model_path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"Face Landmarker model not found: {path}")
        return path

    path = _default_model_path()
    if path.is_file():
        digest = hashlib.md5(path.read_bytes(), usedforsecurity=False).hexdigest()
        if digest == FACE_LANDMARKER_MODEL_MD5:
            return path

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with NamedTemporaryFile(
            mode="wb", dir=path.parent, suffix=".download", delete=False
        ) as model_file:
            temporary_path = Path(model_file.name)
            with urllib.request.urlopen(FACE_LANDMARKER_MODEL_URL, timeout=30) as response:
                while chunk := response.read(1024 * 1024):
                    model_file.write(chunk)

        digest = hashlib.md5(
            temporary_path.read_bytes(), usedforsecurity=False
        ).hexdigest()
        if digest != FACE_LANDMARKER_MODEL_MD5:
            raise RuntimeError("Downloaded Face Landmarker model failed its checksum.")
        temporary_path.replace(path)
    finally:
        if temporary_path and temporary_path.exists():
            temporary_path.unlink()

    return path


def build_face_landmarker(max_faces=10, model_path=None):
    if max_faces < 1:
        raise ValueError("max_faces must be at least 1")
    model_path = _ensure_model(model_path)
    options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_faces=max_faces,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return mp.tasks.vision.FaceLandmarker.create_from_options(options)


def bbox_from_landmarks(face_landmarks, frame_w, frame_h, margin=0.5):
    landmarks = getattr(face_landmarks, "landmark", face_landmarks)
    xs = [lm.x for lm in landmarks]
    ys = [lm.y for lm in landmarks]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    w = x_max - x_min
    h = y_max - y_min
    x_min = max(0.0, x_min - margin * w)
    y_min = max(0.0, y_min - margin * h)
    x_max = min(1.0, x_max + margin * w)
    y_max = min(1.0, y_max + margin * h)
    x0 = math.floor(x_min * frame_w)
    y0 = math.floor(y_min * frame_h)
    x1 = math.ceil(x_max * frame_w)
    y1 = math.ceil(y_max * frame_h)
    return (
        x0,
        y0,
        x1 - x0,
        y1 - y0,
    )


def draw_overlay(frame, box, person_id, state, attentiveness_pct):
    x, y, w, h = box
    color = STATE_COLORS.get(state, (255, 255, 255))
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
    label = f"ID {person_id}: {state}"
    if attentiveness_pct is not None:
        label += f" ({attentiveness_pct}% attentive)"
    cv2.putText(
        frame,
        label,
        (x, max(0, y - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color,
        2,
    )


def draw_summary(frame, counts):
    y0 = 30
    total = sum(counts.values())
    cv2.putText(
        frame,
        f"Total tracked: {total}",
        (10, y0),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )
    for i, (state, count) in enumerate(counts.items()):
        color = STATE_COLORS.get(state, (255, 255, 255))
        cv2.putText(
            frame,
            f"{state}: {count}",
            (10, y0 + 30 * (i + 1)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            color,
            2,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        default=0,
        help="Webcam index or path to a video file (default: 0)",
    )
    parser.add_argument(
        "--log",
        default=None,
        help="Optional path to a CSV file to log per-second summaries.",
    )
    parser.add_argument(
        "--max-faces",
        type=int,
        default=10,
        help="Maximum number of faces to track at once.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Path to a MediaPipe Face Landmarker task model (downloaded on first run if omitted).",
    )
    args = parser.parse_args()
    if args.max_faces < 1:
        parser.error("--max-faces must be at least 1")

    try:
        source = int(args.source)
    except ValueError:
        source = args.source
    if isinstance(source, int) and os.name == "nt":
        cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {source}")

    face_landmarker = None
    csv_writer = None
    csv_file = None
    try:
        face_landmarker = build_face_landmarker(
            max_faces=args.max_faces, model_path=args.model
        )
        tracker = CentroidTracker()
        registry = AttentionRegistry()
        last_log_time = time.time()
        last_timestamp_ms = 0

        if args.log:
            csv_file = open(args.log, mode="w", newline="")
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow([
                "timestamp",
                "attentive",
                "distracted",
                "drowsy",
                "no_face",
                "total",
            ])

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)
            frame_h, frame_w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            timestamp_ms = max(last_timestamp_ms + 1, int(time.monotonic() * 1000))
            results = face_landmarker.detect_for_video(
                mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), timestamp_ms
            )
            last_timestamp_ms = timestamp_ms

            rects = []
            landmarks_by_index = []
            for face_landmarks in results.face_landmarks:
                box = bbox_from_landmarks(face_landmarks, frame_w, frame_h)
                rects.append(box)
                landmarks_by_index.append(face_landmarks)

            tracked = tracker.update(rects)

            box_to_landmarks = dict(zip(rects, landmarks_by_index))
            for person_id, box in tracked.items():
                person = registry.get_or_create(person_id)
                landmarks = box_to_landmarks.get(box)
                if landmarks is None:
                    state = person.mark_no_face()
                else:
                    left_eye_pts = landmarks_to_xy(
                        landmarks, LEFT_EYE, frame_w, frame_h
                    )
                    right_eye_pts = landmarks_to_xy(
                        landmarks, RIGHT_EYE, frame_w, frame_h
                    )
                    mouth_pts = landmarks_to_xy(landmarks, MOUTH, frame_w, frame_h)
                    ear = (
                        eye_aspect_ratio(left_eye_pts)
                        + eye_aspect_ratio(right_eye_pts)
                    ) / 2.0
                    mar = mouth_aspect_ratio(mouth_pts)
                    pose = get_head_pose(landmarks, frame_w, frame_h)
                    yaw, pitch = (pose[0], pose[1]) if pose else (None, None)
                    state = person.update(ear, mar, yaw, pitch)

                draw_overlay(
                    frame, box, person_id, state, person.attentiveness_percentage()
                )

            registry.prune_stale()
            counts = registry.summary()
            draw_summary(frame, counts)

            if csv_writer and time.time() - last_log_time >= 1.0:
                csv_writer.writerow([
                    time.strftime("%Y-%m-%d %H:%M:%S"),
                    counts.get("ATTENTIVE", 0),
                    counts.get("DISTRACTED", 0),
                    counts.get("DROWSY", 0),
                    counts.get("NO FACE", 0),
                    sum(counts.values()),
                ])
                csv_file.flush()
                last_log_time = time.time()

            cv2.imshow("Attention Monitor", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        if face_landmarker is not None:
            face_landmarker.close()
        cv2.destroyAllWindows()
        if csv_file:
            csv_file.close()


if __name__ == "__main__":
    main()