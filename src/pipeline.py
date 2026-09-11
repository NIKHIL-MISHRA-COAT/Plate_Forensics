"""
End-to-end forensic license-plate pipeline.

Stage order (image path):
  detect -> crop -> perspective correct -> denoise -> deblur
          -> super-resolve -> contrast enhance -> OCR

Stage order (video path):
  detect per sampled frame -> track into bursts -> multi-frame fuse per
  burst -> [same per-crop stages as above] -> OCR

Design note on ordering
------------------------
Denoise -> deblur -> super-resolve -> contrast is a deliberate order:
  1. Denoise first so deconvolution (deblur) doesn't amplify sensor noise
     (Wiener/RL deconvolution are noise-sensitive).
  2. Deblur before super-resolution so the SR model receives a sharper,
     more "natural" looking crop closer to its training distribution,
     rather than upscaling blur into larger, harder-to-remove blur.
  3. Super-resolve before contrast enhancement so CLAHE operates at full
     target resolution (contrast/edge enhancement on a tiny crop then
     upscaling would exaggerate blocky artifacts).
  4. Contrast enhancement last, immediately before OCR, since it's the
     step most directly aimed at maximizing character/background
     separation for the recognizer.

Each stage is individually toggleable via config and is only applied if it
improves a simple sharpness+contrast quality metric (quality-guided mode).
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import List, Optional

import cv2
import numpy as np
import yaml

from src.detection.plate_detector import PlateDetector, PlateDetection
from src.enhancement.perspective import correct_perspective
from src.enhancement.denoise import denoise
from src.enhancement.deblur import deblur
from src.enhancement.super_resolution import super_resolve
from src.enhancement.contrast import enhance_contrast
from src.enhancement.multi_frame import enhance_video_crop
from src.ocr.plate_ocr import PlateOCR, OCRResult
from src.video_utils import read_video_frames, group_detections_into_tracks


@dataclass
class PlateResult:
    detection: PlateDetection
    stages: dict                    # name -> np.ndarray, for qualitative before/after
    final_crop: np.ndarray
    ocr: OCRResult
    timing_sec: float


# ---------------------- Quality metrics ---------------------- #

def _gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image.copy()


def _sharpness_score(gray_img: np.ndarray) -> float:
    # Variance of Laplacian as a blur metric
    lap = cv2.Laplacian(gray_img, cv2.CV_64F)
    return float(lap.var())


def _contrast_score(gray_img: np.ndarray) -> float:
    return float(gray_img.std())


def _quality_score(
    image: np.ndarray,
    w_sharp: float = 1.0,
    w_cont: float = 1.0,
) -> float:
    g = _gray(image)
    s = _sharpness_score(g)
    c = _contrast_score(g)
    # Rough normalization to keep scales similar
    s_norm = s / 1e3
    c_norm = c / 50.0
    return w_sharp * s_norm + w_cont * c_norm


# ---------------------- Pipeline ---------------------- #

class ForensicPlatePipeline:
    def __init__(self, config_path: str = "configs/config.yaml"):
        with open(config_path, "r") as f:
            self.cfg = yaml.safe_load(f)

        det_cfg = self.cfg["detection"]
        self.detector = PlateDetector(
            weights=det_cfg["weights"],
            fallback_weights=det_cfg["fallback_weights"],
            conf_threshold=det_cfg["conf_threshold"],
            iou_threshold=det_cfg["iou_threshold"],
        )

        ocr_cfg = self.cfg["ocr"]
        self.ocr = PlateOCR(
            engine=ocr_cfg["engine"],
            languages=ocr_cfg["languages"],
            allowlist=ocr_cfg["allowlist"],
            min_confidence=ocr_cfg["min_confidence"],
            format_patterns=ocr_cfg["format_patterns"],
            gpu=self._has_gpu(),
        )

        # Quality-guided enhancement config
        enh_cfg = self.cfg.get("enhancement", {})
        q_cfg = enh_cfg.get("quality_guidance", {})
        self.q_enabled = q_cfg.get("enabled", True)
        self.q_margin = q_cfg.get("margin", 0.05)  # require >5% improvement
        self.q_weights = q_cfg.get("weights", {"sharpness": 1.0, "contrast": 1.0})

    @staticmethod
    def _has_gpu() -> bool:
        try:
            import torch
            return torch.cuda.is_available()
        except Exception:
            return False

    # ---------------------- Stage gating ---------------------- #

    def _apply_stage_if_better(
        self,
        current: np.ndarray,
        candidate: np.ndarray,
        stage_name: str,
        stages: dict,
    ) -> np.ndarray:
        if not self.q_enabled:
            # Blindly accept all stages (old behavior)
            stages[stage_name] = candidate
            return candidate

        score_before = _quality_score(
            current,
            w_sharp=self.q_weights.get("sharpness", 1.0),
            w_cont=self.q_weights.get("contrast", 1.0),
        )
        score_after = _quality_score(
            candidate,
            w_sharp=self.q_weights.get("sharpness", 1.0),
            w_cont=self.q_weights.get("contrast", 1.0),
        )

        if score_after > score_before * (1.0 + self.q_margin):
            stages[stage_name] = candidate
            return candidate
        else:
            # Skip this stage; keep previous image
            stages[stage_name] = current
            return current

    # ---------------------- Enhancement chain ---------------------- #

    def _enhance_crop(self, crop: np.ndarray, corners: Optional[np.ndarray] = None) -> dict:
        """
        Run the single-crop enhancement chain with quality-guided gating.
        Each stage is only accepted if it improves a simple sharpness+contrast score.
        """
        e = self.cfg["enhancement"]
        stages = {"00_input_crop": crop}
        current = crop

        # Perspective correction
        if e.get("perspective", {}).get("enabled", False):
            candidate = correct_perspective(
                current,
                target_size=(
                    e["perspective"]["target_width"],
                    e["perspective"]["target_height"],
                ),
                corners=corners,
            )
            current = self._apply_stage_if_better(current, candidate, "01_perspective_corrected", stages)
        else:
            stages["01_perspective_corrected"] = current

        # Denoise
        if e.get("denoise", {}).get("enabled", False):
            cfg = e["denoise"]

            # Build kwargs, renaming if needed
            kwargs = {
                k: v
                for k, v in cfg.items()
                if k not in ("method", "enabled")
            }

            # Rename keys if your denoise() uses snake_case
            if "templateWindowSize" in kwargs:
                kwargs["template_window_size"] = kwargs.pop("templateWindowSize")
            if "searchWindowSize" in kwargs:
                kwargs["search_window_size"] = kwargs.pop("searchWindowSize")

            candidate = denoise(current, method=cfg["method"], **kwargs)
            current = self._apply_stage_if_better(current, candidate, "02_denoised", stages)
        else:
            stages["02_denoised"] = current

        # Deblur
        if e.get("deblur", {}).get("enabled", False):
            candidate = deblur(
                current,
                **{
                    k: v
                    for k, v in e["deblur"].items()
                    if k not in ("method", "enabled")
                },
                method=e["deblur"]["method"],
            )
            current = self._apply_stage_if_better(current, candidate, "03_deblurred", stages)
        else:
            stages["03_deblurred"] = current

        # Super-resolution
        if e.get("super_resolution", {}).get("enabled", False):
            sr_cfg = e["super_resolution"]
            candidate = super_resolve(
                current,
                method=sr_cfg["method"],
                scale=sr_cfg["scale"],
                model_name=sr_cfg["model_name"],
                tile=sr_cfg["tile"],
                device="cuda" if self._has_gpu() else "cpu",
            )
            current = self._apply_stage_if_better(current, candidate, "04_super_resolved", stages)
        else:
            stages["04_super_resolved"] = current

        # Contrast enhancement
        if e.get("contrast", {}).get("enabled", False):
            candidate = enhance_contrast(
                current,
                method=e["contrast"]["method"],
                clip_limit=e["contrast"]["clip_limit"],
                tile_grid_size=e["contrast"]["tile_grid_size"],
            )
            current = self._apply_stage_if_better(current, candidate, "05_contrast_enhanced", stages)
        else:
            stages["05_contrast_enhanced"] = current

        return stages

    # ---------------------- Image processing ---------------------- #

    def process_image(self, image: np.ndarray) -> List[PlateResult]:
        t0 = time.time()
        detections = self.detector.detect(image)
        results = []

        for det in detections:
            crop = det.crop(image)
            if crop.size == 0:
                continue

            stages = self._enhance_crop(crop, corners=det.corners)
            final_crop = list(stages.values())[-1]
            ocr_result = self.ocr.read(final_crop)

            results.append(PlateResult(
                detection=det,
                stages=stages,
                final_crop=final_crop,
                ocr=ocr_result,
                timing_sec=time.time() - t0,
            ))

        return results

    def process_image_path(self, image_path: str) -> List[PlateResult]:
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")
        return self.process_image(image)

    # ---------------------- Video processing ---------------------- #

    def process_video(self, video_path: str, frame_stride: int = 2) -> List[PlateResult]:
        mf_cfg = self.cfg["enhancement"]["multi_frame"]
        frames = read_video_frames(video_path, stride=frame_stride)
        if not frames:
            return []

        per_frame_detections = [self.detector.detect(f) for f in frames]

        if mf_cfg["enabled"]:
            tracks = group_detections_into_tracks(per_frame_detections)
            results = []
            t0 = time.time()

            for track in tracks:
                crops = [
                    det.crop(frames[fi])
                    for fi, det in track
                    if det.crop(frames[fi]).size > 0
                ]
                if not crops:
                    continue

                fused_crop = enhance_video_crop(
                    crops,
                    max_frames=mf_cfg["max_frames"],
                    sharpness_percentile=mf_cfg["sharpness_percentile"],
                    alignment=mf_cfg["alignment"],
                    fusion=mf_cfg["fusion"],
                )

                best_det = max((det for _, det in track), key=lambda d: d.confidence)
                stages = self._enhance_crop(fused_crop, corners=best_det.corners)
                stages = {"00a_multi_frame_fused": fused_crop, **stages}

                final_crop = list(stages.values())[-1]
                ocr_result = self.ocr.read(final_crop)

                results.append(PlateResult(
                    detection=best_det,
                    stages=stages,
                    final_crop=final_crop,
                    ocr=ocr_result,
                    timing_sec=time.time() - t0,
                ))

            return results

        # Multi-frame fusion disabled: process every detected frame independently
        results = []
        for frame, detections in zip(frames, per_frame_detections):
            results.extend(self.process_image(frame))
        return results

    # ---------------------- Visualization ---------------------- #

    def save_visual_report(self, result: PlateResult, out_dir: str, prefix: str = "plate"):
        os.makedirs(out_dir, exist_ok=True)

        for name, img in result.stages.items():
            cv2.imwrite(os.path.join(out_dir, f"{prefix}_{name}.png"), img)

        keys = list(result.stages.keys())
        first, last = result.stages[keys[0]], result.stages[keys[-1]]
        h = max(first.shape[0], last.shape[0])

        first_r = cv2.resize(first, (int(first.shape[1] * h / first.shape[0]), h))
        last_r = cv2.resize(last, (int(last.shape[1] * h / last.shape[0]), h))

        strip = np.hstack([first_r, np.full((h, 10, 3), 255, np.uint8), last_r])
        cv2.imwrite(os.path.join(out_dir, f"{prefix}_before_after.png"), strip)