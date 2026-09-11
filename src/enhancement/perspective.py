"""
Perspective / geometric distortion correction.

Design choice
-------------
Plates photographed off-axis are foreshortened and skewed. Rather than
training a dedicated keypoint model (expensive to annotate), we use a
classical, dependency-light approach that works well once we already have a
tight plate crop from the detector:

  1. Threshold + find the largest 4-point contour inside the crop (the
     plate's outer rim is usually the dominant rectangle after the
     background has mostly been cropped away by the detector).
  2. If a good quadrilateral isn't found (e.g. very shallow angle, glare),
     fall back to treating the crop's own bounding box as the "corners" --
     i.e. skip correction rather than warp something wrong.
  3. Warp the 4 points to a canonical front-facing rectangle
     (target_width x target_height) with cv2.getPerspectiveTransform.

This keeps the step model-free and fast, and is easy to swap for a
keypoint-regression network later if the fine-tuned detector is extended to
predict 4 corner points directly (noted in the README as a future upgrade).
"""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np


def _order_points(pts: np.ndarray) -> np.ndarray:
    """Order 4 points as top-left, top-right, bottom-right, bottom-left."""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def find_plate_quadrilateral(crop: np.ndarray) -> Optional[np.ndarray]:
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop.copy()
    gray = cv2.bilateralFilter(gray, 9, 75, 75)
    edges = cv2.Canny(gray, 40, 150)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    crop_area = crop.shape[0] * crop.shape[1]
    best_quad, best_area = None, 0

    for c in sorted(contours, key=cv2.contourArea, reverse=True)[:10]:
        area = cv2.contourArea(c)
        if area < 0.25 * crop_area:      # reject tiny/noisy contours
            continue
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4 and area > best_area:
            best_quad = approx.reshape(4, 2)
            best_area = area

    return best_quad


def correct_perspective(
    crop: np.ndarray,
    target_size: Tuple[int, int] = (300, 100),
    corners: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Warp `crop` to a front-facing rectangle of `target_size` = (width, height).
    If `corners` (4,2) are supplied (e.g. from a keypoint model) they are
    used directly; otherwise we attempt classical quadrilateral detection.
    Falls back to a plain resize if no reliable quadrilateral is found.
    """
    target_w, target_h = target_size

    quad = corners if corners is not None else find_plate_quadrilateral(crop)
    if quad is None:
        return cv2.resize(crop, (target_w, target_h), interpolation=cv2.INTER_CUBIC)

    src = _order_points(quad.astype("float32"))
    dst = np.array(
        [[0, 0], [target_w - 1, 0], [target_w - 1, target_h - 1], [0, target_h - 1]],
        dtype="float32",
    )
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(crop, M, (target_w, target_h))
    return warped
