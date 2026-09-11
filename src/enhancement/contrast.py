"""
Contrast / brightness enhancement.

Design choice
-------------
CLAHE (Contrast Limited Adaptive Histogram Equalization), applied on the L
channel in LAB space, is the default: it boosts local contrast (helpful for
faint/washed-out plates under harsh headlight glare or heavy shadow) while
the "contrast limit" prevents the noise amplification that plain histogram
equalization causes -- important because we denoise *before* this step, and
a naive global equalization would re-introduce noise-like artifacts right
before OCR.
"""

from __future__ import annotations

import cv2
import numpy as np


def clahe_enhance(image: np.ndarray, clip_limit: float = 2.0, tile_grid_size=(8, 8)) -> np.ndarray:
    if image.ndim == 2:
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tuple(tile_grid_size))
        return clahe.apply(image)

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tuple(tile_grid_size))
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)


def hist_eq(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return cv2.equalizeHist(image)
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    y = cv2.equalizeHist(y)
    return cv2.cvtColor(cv2.merge((y, cr, cb)), cv2.COLOR_YCrCb2BGR)


def enhance_contrast(image: np.ndarray, method: str = "clahe", clip_limit: float = 2.0,
                      tile_grid_size=(8, 8)) -> np.ndarray:
    if method == "none":
        return image
    if method == "clahe":
        return clahe_enhance(image, clip_limit, tile_grid_size)
    if method == "hist_eq":
        return hist_eq(image)
    raise ValueError(f"Unknown contrast method: {method}")
