"""
CLI entry point.

Usage:
    python scripts/run_pipeline.py --input path/to/image_or_video --output outputs/run1
    python scripts/run_pipeline.py --input path/to/video.mp4 --output outputs/run1 --stride 2

For each detected plate, writes:
    outputs/run1/plate_00_<stage>.png   (every enhancement stage, for qualitative analysis)
    outputs/run1/plate_00_before_after.png
    outputs/run1/results.json           (bbox, confidence, OCR text, OCR confidence, timing)
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.pipeline import ForensicPlatePipeline  # noqa: E402


def is_video(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in {".mp4", ".avi", ".mov", ".mkv"}


def main():
    parser = argparse.ArgumentParser(description="Forensic license-plate enhancement & recovery")
    parser.add_argument("--input", required=True, help="Path to an image or video file")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--stride", type=int, default=2, help="Frame stride for video processing")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    pipeline = ForensicPlatePipeline(config_path=args.config)

    if is_video(args.input):
        results = pipeline.process_video(args.input, frame_stride=args.stride)
    else:
        results = pipeline.process_image_path(args.input)

    if not results:
        print("No plates detected.")
        return

    summary = []
    for i, r in enumerate(results):
        prefix = f"plate_{i:02d}"
        pipeline.save_visual_report(r, args.output, prefix=prefix)
        summary.append({
            "plate_index": i,
            "bbox": list(r.detection.bbox),
            "detection_confidence": r.detection.confidence,
            "ocr_text": r.ocr.text,
            "ocr_confidence": r.ocr.confidence,
            "matches_known_format": r.ocr.matches_known_format,
            "matched_pattern": r.ocr.matched_pattern,
            "timing_sec": r.timing_sec,
        })
        print(f"[{prefix}] text='{r.ocr.text}' conf={r.ocr.confidence:.2f} "
              f"bbox={r.detection.bbox} det_conf={r.detection.confidence:.2f}")

    with open(os.path.join(args.output, "results.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved {len(results)} result(s) to {args.output}")


if __name__ == "__main__":
    main()
