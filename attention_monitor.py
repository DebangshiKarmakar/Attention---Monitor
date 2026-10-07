""""Main entry point for the attention monitor application.

Works for both single webcam and classroom or office feeds with multiple people.
It detects every face independently.
Usage:
    python attention_monitor.py       # webcam
Press 'q' to quit.
"""

import argparse
import csv
import time

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


def build_face_mesh(max_faces=10):
    mp_face_mesh = mp.solutions.face_mesh
    return mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=max_faces,
        refine_landmarks=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )


def bbox_from_landmarks(face_landmarks, frame_w, frame_h, margin=0.5):
    xs = [lm.x for lm in face_landmarks.landmark]
    ys = [lm.y for lm in face_landmarks.landmark]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    w = x_max - x_min
    h = y_max - y_min
    x_min = max(0.0, x_min - margin * w)
    y_min = max(0.0, y_min - margin * h)
    x_max = min(1.0, x_max + margin * w)
    y_max = min(1.0, y_max + margin * h)
    return (
        int(x_min * frame_w),
        int(y_min * frame_h),
        int((x_max - x_min) * frame_w),
        int((y_max - y_min) * frame_h),
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
    args = parser.parse_args()

    source = int(args.source) if str(args.source).isdigit() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {source}")

    face_mesh = build_face_mesh(max_faces=args.max_faces)
    tracker = CentroidTracker()
    registry = AttentionRegistry()
    csv_writer = None
    csv_file = None
    last_log_time = time.time()

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
        results = face_mesh.process(rgb)

        rects = []
        landmarks_by_index = []
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                box = bbox_from_landmarks(face_landmarks, frame_w, frame_h)
                rects.append(box)
                landmarks_by_index.append(face_landmarks)

        tracked = tracker.update(rects)

        box_to_landmarks = {}
        for box, landmarks in zip(rects, landmarks_by_index):
            box_to_landmarks[box] = landmarks

        for person_id, box in tracked.items():
            person = registry.get_or_create(person_id)
            landmarks = box_to_landmarks.get(box)
            if landmarks is None:
                state = person.mark_no_face()
            else:
                left_eye_pts = landmarks_to_xy(landmarks, LEFT_EYE, frame_w, frame_h)
                right_eye_pts = landmarks_to_xy(landmarks, RIGHT_EYE, frame_w, frame_h)
                mouth_pts = landmarks_to_xy(landmarks, MOUTH, frame_w, frame_h)
                ear = (
                    eye_aspect_ratio(left_eye_pts) + eye_aspect_ratio(right_eye_pts)
                ) / 2.0
                mar = mouth_aspect_ratio(mouth_pts)
                pose = get_head_pose(landmarks, frame_w, frame_h)
                yaw, pitch = (pose[0], pose[1]) if pose else (None, None)
                state = person.update(ear, mar, yaw, pitch)

            draw_overlay(frame, box, person_id, state, person.attentiveness_percentage())

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

    cap.release()
    cv2.destroyAllWindows()
    if csv_file:
        csv_file.close()


if __name__ == "__main__":
    main()