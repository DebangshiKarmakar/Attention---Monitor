import math

import numpy as np

LEFT_EYE = [33, 133, 159, 145, 153, 144]
RIGHT_EYE = [362, 382, 381, 380, 263, 373]
MOUTH = [61, 291, 0, 17, 14, 78, 308]


def landmarks_to_xy(landmarks, indices, frame_w, frame_h):
    """Return a 2D array of normalized landmark coordinates in pixel space."""
    points = []
    for index in indices:
        mark = landmarks.landmark[index]
        points.append((mark.x * frame_w, mark.y * frame_h))
    return np.asarray(points, dtype=np.float32)


def eye_aspect_ratio(eye_points):
    """Compute a rough eye-aspect-ratio based on eye landmark geometry."""
    if len(eye_points) < 6:
        return 0.0

    p1, p2, p3, p4, p5, p6 = eye_points
    vertical = np.linalg.norm(p2 - p6) + np.linalg.norm(p3 - p5)
    horizontal = np.linalg.norm(p1 - p4)
    if horizontal < 1e-6:
        return 0.0
    return vertical / (2.0 * horizontal)


def mouth_aspect_ratio(mouth_points):
    """Compute a rough mouth opening ratio."""
    if len(mouth_points) < 4:
        return 0.0

    left = mouth_points[0]
    right = mouth_points[1]
    top = mouth_points[2]
    bottom = mouth_points[3]
    width = np.linalg.norm(left - right)
    height = np.linalg.norm(top - bottom)
    if width < 1e-6:
        return 0.0
    return height / width


def get_head_pose(landmarks, frame_w, frame_h):
    """Estimate a simple yaw/pitch from the current face geometry."""
    nose = landmarks.landmark[1]
    left_eye = landmarks.landmark[33]
    right_eye = landmarks.landmark[263]
    eye_mid_x = (left_eye.x + right_eye.x) / 2.0
    eye_mid_y = (left_eye.y + right_eye.y) / 2.0

    eye_span = right_eye.x - left_eye.x
    yaw = math.degrees(math.atan2(eye_span, 0.25))
    pitch = math.degrees(
        math.atan2(nose.y - eye_mid_y, max(0.05, abs(nose.x - eye_mid_x)))
    )
    return (yaw, pitch)
