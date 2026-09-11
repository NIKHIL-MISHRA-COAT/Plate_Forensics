"""
License plate detector.

Design choice
-------------
We use YOLOv8 (ultralytics) as the detector backbone:
  - Single-class detection (plate / no-plate) is a well-posed problem for a
    small-to-medium YOLOv8 model (n/s), so we get good accuracy with a light,
    fast model that can run per-frame on video without being the bottleneck.
  - A large ecosystem of open license-plate datasets (CCPD, OpenALPR
    benchmark, Roboflow "license-plate-recognition" sets) already provide
    YOLO-format annotations, so fine-tuning is a matter of pointing at a
    data.yaml (see notebooks/01_finetune_plate_detector.ipynb).
  - If no fine-tuned weights are present (e.g. first run before training),
    we fall back to a generic pretrained YOLOv8 model + a Haar-cascade plate
    detector as an OpenCV-only safety net, so the pipeline is never blocked
    on network access to download fine-tuned weights.

Output contract
----------------
detect(image) -> List[PlateDetection]
    Each PlateDetection carries a bbox (x1, y1, x2, y2), a confidence score,
    and (optionally) 4 corner points if a quadrilateral/keypoint model is
    used -- corners are used downstream for perspective correction.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np
# COCO class indices for vehicle types. Used only when we've fallen back to
# a generic COCO-pretrained YOLO model (i.e. no fine-tuned plate weights are
# available) -- in that mode YOLO boxes are vehicles, not plates, and we use
# them solely to narrow where we look for a plate.
COCO_VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

@dataclass
class PlateDetection:
    bbox: Tuple[int, int, int, int]        # x1, y1, x2, y2 in original image coords
    confidence: float
    corners: Optional[np.ndarray] = None    # (4, 2) array if available, else None

    def crop(self, image: np.ndarray, pad_ratio: float = 0.08) -> np.ndarray:
        h, w = image.shape[:2]
        x1, y1, x2, y2 = self.bbox
        bw, bh = x2 - x1, y2 - y1
        px, py = int(bw * pad_ratio), int(bh * pad_ratio)
        x1, y1 = max(0, x1 - px), max(0, y1 - py)
        x2, y2 = min(w, x2 + px), min(h, y2 + py)
        return image[y1:y2, x1:x2].copy()


class PlateDetector:
    def __init__(
        self,
        weights: str = "models/plate_detector/best.pt",
        fallback_weights: str = "yolov8n.pt",
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
    ):
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self._backend = None
        self._model = None
        self._haar = None
        # True only when `weights` points at a real, fine-tuned single-class
        # plate model. False for the generic COCO fallback, which changes
        # how we interpret the model's boxes in detect().
        self._is_finetuned_plate_model = False

        weights_to_try = weights if os.path.exists(weights) else None

        if weights_to_try is not None:
            self._load_yolo(weights_to_try)
            self._is_finetuned_plate_model = True
        else:
            print(
                f"[PlateDetector] Fine-tuned weights not found at '{weights}'. "
                f"Falling back to: generic pretrained '{fallback_weights}' for "
                f"vehicle localization -> Haar cascade for plate localization "
                f"within each vehicle. Run notebooks/01_finetune_plate_detector.ipynb "
                f"to train a plate-specific model for production accuracy."
            )
            self._load_haar()  # always available, needed either standalone or as stage 2
            try:
                self._load_yolo(fallback_weights)
                self._is_finetuned_plate_model = False
            except Exception as e:  # ultralytics weights unavailable (no network)
                print(f"[PlateDetector] Could not load YOLO fallback ({e}). "
                      f"Using OpenCV Haar-cascade plate detector on the full frame instead.")
                self._backend = "haar"  # no vehicle-localization stage available

    # ------------------------------------------------------------------ #
        # ------------------------------------------------------------------ #
    def _load_yolo(self, weights_path: str):
        from ultralytics import YOLO  # local import: heavy, optional dependency
        self._model = YOLO(weights_path)
        self._backend = "yolo"

    def _load_haar(self):
        # OpenCV ships a Russian-plate Haar cascade that generalizes
        # reasonably to rectangular plates of similar aspect ratio. This is
        # intentionally a low-accuracy, zero-dependency fallback so the
        # pipeline degrades gracefully rather than crashing.
        cascade_path = cv2.data.haarcascades + "haarcascade_russian_plate_number.xml"
        self._haar = cv2.CascadeClassifier(cascade_path)
        if self._backend is None:
            self._backend = "haar"

    # ------------------------------------------------------------------ #
    def detect(self, image: np.ndarray) -> List[PlateDetection]:
        if self._backend == "yolo" and self._is_finetuned_plate_model:
            return self._detect_yolo_plates(image)
        elif self._backend == "yolo" and not self._is_finetuned_plate_model:
            return self._detect_vehicle_then_plate(image)
        elif self._backend == "haar":
            return self._detect_haar(image, offset=(0, 0))
        raise RuntimeError("No detection backend initialized.")

    def _detect_yolo_plates(self, image: np.ndarray) -> List[PlateDetection]:
        """Fine-tuned single-class plate model: every box IS a plate."""
        results = self._model.predict(
            image, conf=self.conf_threshold, iou=self.iou_threshold, verbose=False
        )
        detections: List[PlateDetection] = []
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                conf = float(box.conf[0].cpu().numpy())
                detections.append(PlateDetection(bbox=(x1, y1, x2, y2), confidence=conf))
        return detections

    def _detect_vehicle_then_plate(self, image: np.ndarray) -> List[PlateDetection]:
        """
        Generic COCO model: boxes are vehicles, not plates. Localize
        vehicles first, then run the Haar cascade inside each vehicle crop
        to find the actual plate, translating coordinates back to the
        original image.
        """
        results = self._model.predict(
            image, conf=self.conf_threshold, iou=self.iou_threshold, verbose=False
        )
        h, w = image.shape[:2]
        vehicle_boxes = []
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                cls_id = int(box.cls[0].cpu().numpy())
                if cls_id not in COCO_VEHICLE_CLASSES:
                    continue
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                vehicle_boxes.append((x1, y1, x2, y2))

        detections: List[PlateDetection] = []
        if vehicle_boxes:
            for (vx1, vy1, vx2, vy2) in vehicle_boxes:
                vx1, vy1 = max(0, vx1), max(0, vy1)
                vx2, vy2 = min(w, vx2), min(h, vy2)
                vehicle_crop = image[vy1:vy2, vx1:vx2]
                if vehicle_crop.size == 0:
                    continue
                detections.extend(self._detect_haar(vehicle_crop, offset=(vx1, vy1)))
        else:
            # No vehicle found (e.g. plate already fills the frame, or a
            # non-vehicle framing) -- fall back to scanning the full image.
            detections.extend(self._detect_haar(image, offset=(0, 0)))
        return detections

    def _detect_haar(self, image: np.ndarray, offset: Tuple[int, int] = (0, 0)) -> List[PlateDetection]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        boxes = self._haar.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=4, minSize=(60, 20))
        ox, oy = offset
        detections = []
        for (x, y, w, h) in boxes:
            # Haar gives no confidence; assign a flat, conservative score so
            # downstream OCR-confidence fusion still makes sense.
            detections.append(PlateDetection(
                bbox=(x + ox, y + oy, x + w + ox, y + h + oy), confidence=0.5
            ))
        return detections