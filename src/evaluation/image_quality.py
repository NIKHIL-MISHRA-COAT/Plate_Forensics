"""
Image quality metrics: PSNR / SSIM.

Design choice
-------------
PSNR and SSIM are reported as *before vs. after* comparisons against the
original degraded crop (not against an idealized "clean" reference, which
we don't have for real forensic footage) to quantify how much each
enhancement stage changes the signal, plus, where synthetic degraded/clean
pairs exist (i.e. our fine-tuning validation set, see
notebooks/02_finetune_super_resolution.ipynb), PSNR/SSIM against the true
clean image -- this is the more meaningful use of these metrics, since
"enhanced vs. original-degraded" alone doesn't guarantee the enhancement
moved *towards* the ground truth rather than just changing the image.
We therefore always pair image-quality metrics with the OCR-accuracy
metrics in evaluation/ocr_accuracy.py, since OCR correctness is the metric
that actually matters for this task.
"""

from __future__ import annotations

import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as sk_psnr
from skimage.metrics import structural_similarity as sk_ssim


def _match_size(a: np.ndarray, b: np.ndarray) -> tuple:
    h, w = min(a.shape[0], b.shape[0]), min(a.shape[1], b.shape[1])
    return cv2.resize(a, (w, h)), cv2.resize(b, (w, h))


def compute_psnr(img_a: np.ndarray, img_b: np.ndarray) -> float:
    img_a, img_b = _match_size(img_a, img_b)
    return float(sk_psnr(img_a, img_b, data_range=255))


def compute_ssim(img_a: np.ndarray, img_b: np.ndarray) -> float:
    img_a, img_b = _match_size(img_a, img_b)
    gray_a = cv2.cvtColor(img_a, cv2.COLOR_BGR2GRAY) if img_a.ndim == 3 else img_a
    gray_b = cv2.cvtColor(img_b, cv2.COLOR_BGR2GRAY) if img_b.ndim == 3 else img_b
    return float(sk_ssim(gray_a, gray_b, data_range=255))


def compute_quality_report(reference: np.ndarray, comparison: np.ndarray) -> dict:
    return {
        "psnr": compute_psnr(reference, comparison),
        "ssim": compute_ssim(reference, comparison),
    }
