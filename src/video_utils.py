"""
Video I/O helpers: frame extraction and simple IoU-based plate tracking so
that a burst of frames can be associated with the same physical plate for
multi-frame fusion (src/enhancement/multi_frame.py).
"""

from __future__ import annotations

from typing import List, Tuple

import cv2
import numpy as np

from src.detection.plate_detector import PlateDetection


def read_video_frames(video_path: str, stride: int = 1, max_frames: int = 300) -> List[np.ndarray]:
    cap = cv2.VideoCapture(video_path)
    frames = []
    idx = 0
    while cap.isOpened() and len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % stride == 0:
            frames.append(frame)
        idx += 1
    cap.release()
    return frames


def iou(box_a: Tuple[int, int, int, int], box_b: Tuple[int, int, int, int]) -> float:
    xa1, ya1, xa2, ya2 = box_a
    xb1, yb1, xb2, yb2 = box_b
    inter_x1, inter_y1 = max(xa1, xb1), max(ya1, yb1)
    inter_x2, inter_y2 = min(xa2, xb2), min(ya2, yb2)
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    area_a = (xa2 - xa1) * (ya2 - ya1)
    area_b = (xb2 - xb1) * (yb2 - yb1)
    union = area_a + area_b - inter_area
    return inter_area / union if union > 0 else 0.0


def group_detections_into_tracks(
    per_frame_detections: List[List[PlateDetection]],
    iou_threshold: float = 0.3,
) -> List[List[Tuple[int, PlateDetection]]]:
    """
    Very lightweight greedy IoU tracker: sufficient for short bursts where a
    vehicle's plate moves smoothly across consecutive frames. Returns a list
    of tracks; each track is a list of (frame_index, PlateDetection).
    For a production system with occlusion/re-identification needs, swap in
    ByteTrack/DeepSORT.
    """
    tracks: List[List[Tuple[int, PlateDetection]]] = []

    for frame_idx, detections in enumerate(per_frame_detections):
        for det in detections:
            matched = False
            for track in tracks:
                last_frame_idx, last_det = track[-1]
                if frame_idx - last_frame_idx <= 3 and iou(det.bbox, last_det.bbox) >= iou_threshold:
                    track.append((frame_idx, det))
                    matched = True
                    break
            if not matched:
                tracks.append([(frame_idx, det)])

    return tracks
