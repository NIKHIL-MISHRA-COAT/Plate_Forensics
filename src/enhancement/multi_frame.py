"""
Video-specific enhancement: stabilization + multi-frame fusion.

Design choice
-------------
A single video frame of a moving vehicle is often the *worst* frame to OCR
(motion blur peaks mid-frame). Instead, we exploit temporal redundancy:

  1. Select a pool of the sharpest frames from the plate's tracked crop
     region (Laplacian-variance is a cheap, effective sharpness proxy).
  2. Align them to a common reference with ECC (Enhanced Correlation
     Coefficient) affine alignment -- robust to small illumination changes,
     which is common frame-to-frame in video -- or ORB+homography for
     larger viewpoint changes.
  3. Fuse via per-pixel median (robust to outlier misalignment / occlusion)
     or mean (slightly sharper when alignment is reliable).

This is essentially a lightweight, targeted multi-frame super-resolution:
it doesn't need a trained model and directly attacks the two dominant video
degradations (jitter and transient motion blur) before the per-frame
enhancement stages (deblur / SR / contrast) run on the fused result.

Note: this module assumes a fixed detection box is reused for a short burst
of frames (typical for a passing vehicle across ~0.3-0.5s); the pipeline
handles re-detecting periodically for longer sequences.
"""

from __future__ import annotations

from typing import List

import cv2
import numpy as np


def sharpness_score(image: np.ndarray) -> float:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def select_sharp_frames(frames: List[np.ndarray], max_frames: int = 12,
                         sharpness_percentile: float = 60) -> List[np.ndarray]:
    if len(frames) <= max_frames:
        scores = [sharpness_score(f) for f in frames]
    else:
        scores = [sharpness_score(f) for f in frames]

    threshold = np.percentile(scores, sharpness_percentile) if scores else 0
    kept = [f for f, s in zip(frames, scores) if s >= threshold]
    kept = sorted(kept, key=sharpness_score, reverse=True)[:max_frames]
    return kept if kept else frames[:1]


def _align_ecc(reference: np.ndarray, moving: np.ndarray) -> np.ndarray:
    ref_gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY).astype(np.float32)
    mov_gray = cv2.cvtColor(moving, cv2.COLOR_BGR2GRAY).astype(np.float32)
    warp_matrix = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 1e-4)
    try:
        _, warp_matrix = cv2.findTransformECC(
            ref_gray, mov_gray, warp_matrix, cv2.MOTION_AFFINE, criteria
        )
        aligned = cv2.warpAffine(
            moving, warp_matrix, (reference.shape[1], reference.shape[0]),
            flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
        )
        return aligned
    except cv2.error:
        # Alignment failed (e.g. too little texture) -- fall back to raw frame,
        # resized to match reference dims.
        return cv2.resize(moving, (reference.shape[1], reference.shape[0]))


def _align_orb(reference: np.ndarray, moving: np.ndarray) -> np.ndarray:
    orb = cv2.ORB_create(500)
    ref_gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
    mov_gray = cv2.cvtColor(moving, cv2.COLOR_BGR2GRAY)
    kp1, des1 = orb.detectAndCompute(ref_gray, None)
    kp2, des2 = orb.detectAndCompute(mov_gray, None)
    if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
        return cv2.resize(moving, (reference.shape[1], reference.shape[0]))

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = sorted(matcher.match(des1, des2), key=lambda m: m.distance)[:50]
    if len(matches) < 4:
        return cv2.resize(moving, (reference.shape[1], reference.shape[0]))

    src_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    if H is None:
        return cv2.resize(moving, (reference.shape[1], reference.shape[0]))
    return cv2.warpPerspective(moving, H, (reference.shape[1], reference.shape[0]))


def fuse_frames(frames: List[np.ndarray], alignment: str = "ecc", fusion: str = "median") -> np.ndarray:
    if len(frames) == 1:
        return frames[0]

    reference = max(frames, key=sharpness_score)
    ref_h, ref_w = reference.shape[:2]
    resized_ref = reference

    aligner = _align_ecc if alignment == "ecc" else _align_orb
    aligned = [resized_ref]
    for f in frames:
        if f is reference:
            continue
        f_resized = cv2.resize(f, (ref_w, ref_h)) if f.shape[:2] != (ref_h, ref_w) else f
        aligned.append(aligner(resized_ref, f_resized))

    stack = np.stack(aligned, axis=0).astype(np.float32)
    fused = np.median(stack, axis=0) if fusion == "median" else np.mean(stack, axis=0)
    return np.clip(fused, 0, 255).astype(np.uint8)


def enhance_video_crop(
    frames: List[np.ndarray],
    max_frames: int = 12,
    sharpness_percentile: float = 60,
    alignment: str = "ecc",
    fusion: str = "median",
) -> np.ndarray:
    sharp = select_sharp_frames(frames, max_frames, sharpness_percentile)
    return fuse_frames(sharp, alignment, fusion)
