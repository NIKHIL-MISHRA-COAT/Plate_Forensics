"""
Denoising.

Design choice
-------------
Sensor noise and compression (block) artifacts are common in CCTV/dashcam
footage. We default to OpenCV's fastNlMeansDenoisingColored, which handles
both Gaussian sensor noise and mild compression artifacts well without
over-smoothing character edges (important -- OCR cares about edges more
than perceptual smoothness). Bilateral filtering is offered as a faster,
edge-preserving alternative for large batch/video jobs where speed matters
more than best-case quality.

We deliberately do NOT reach for a learned denoiser here: classical
denoising is already near ceiling for this sub-problem on typical
compression noise, and keeping this step cheap leaves more of the pipeline's
model budget for detection, deblurring, and super-resolution where classical
methods fall further short of learned ones.
"""

from __future__ import annotations

import cv2
import numpy as np


def denoise(
    image: np.ndarray,
    method: str = "fastNlMeans",
    h: int = 10,
    template_window_size: int = 7,
    search_window_size: int = 21,
) -> np.ndarray:
    if method == "none":
        return image
    if method == "bilateral":
        return cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)
    if method == "fastNlMeans":
        if image.ndim == 3:
            return cv2.fastNlMeansDenoisingColored(
                image, None, h, h, template_window_size, search_window_size
            )
        return cv2.fastNlMeansDenoising(
            image, None, h, template_window_size, search_window_size
        )
    raise ValueError(f"Unknown denoise method: {method}")
