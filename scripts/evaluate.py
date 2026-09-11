"""
Batch evaluation against a labeled dataset.

Expected dataset layout (see data/raw/README.md):

    data/raw/
      images/
        img001.jpg
        img002.jpg
      annotations.csv        # columns: filename, plate_text  (one row per plate)

Usage:
    python scripts/evaluate.py --data_dir data/raw --output outputs/eval_run1

Produces:
    outputs/eval_run1/metrics.json         -- PSNR/SSIM + CER/exact-match summary
    outputs/eval_run1/per_sample.csv       -- row per sample, for the report's failure analysis
    outputs/eval_run1/visual_grids/        -- before/after qualitative panels (if enabled)
"""

import argparse
import csv
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import cv2  # noqa: E402

from src.pipeline import ForensicPlatePipeline  # noqa: E402
from src.evaluation.image_quality import compute_quality_report  # noqa: E402
from src.evaluation.ocr_accuracy import evaluate_dataset  # noqa: E402


def load_annotations(csv_path: str):
    rows = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description="Evaluate pipeline against labeled data")
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    grids_dir = os.path.join(args.output, "visual_grids")
    os.makedirs(grids_dir, exist_ok=True)

    annotations = load_annotations(os.path.join(args.data_dir, "annotations.csv"))
    pipeline = ForensicPlatePipeline(config_path=args.config)

    predictions, ground_truths, per_sample_rows = [], [], []
    quality_scores = []

    for row in annotations:
        img_path = os.path.join(args.data_dir, "images", row["filename"])
        gt_text = row["plate_text"]

        image = cv2.imread(img_path)
        if image is None:
            print(f"WARNING: could not read {img_path}, skipping")
            continue

        results = pipeline.process_image(image)
        if not results:
            predictions.append("")
            ground_truths.append(gt_text)
            per_sample_rows.append({"filename": row["filename"], "gt": gt_text,
                                     "pred": "", "detected": False})
            continue

        # Take the highest-confidence detection as the "main" plate for
        # single-plate-per-image evaluation datasets.
        best = max(results, key=lambda r: r.detection.confidence)
        predictions.append(best.ocr.text)
        ground_truths.append(gt_text)

        quality = compute_quality_report(best.stages["00_input_crop"], best.final_crop)
        quality_scores.append(quality)

        per_sample_rows.append({
            "filename": row["filename"], "gt": gt_text, "pred": best.ocr.text,
            "detected": True, "ocr_confidence": best.ocr.confidence,
            "psnr": quality["psnr"], "ssim": quality["ssim"],
        })

        pipeline.save_visual_report(best, grids_dir, prefix=os.path.splitext(row["filename"])[0])

    ocr_metrics = evaluate_dataset(predictions, ground_truths)
    avg_psnr = sum(q["psnr"] for q in quality_scores) / len(quality_scores) if quality_scores else 0.0
    avg_ssim = sum(q["ssim"] for q in quality_scores) / len(quality_scores) if quality_scores else 0.0

    metrics = {
        "n_samples": len(annotations),
        "n_detected": sum(1 for r in per_sample_rows if r.get("detected")),
        "exact_match_accuracy": ocr_metrics["exact_match_accuracy"],
        "mean_cer": ocr_metrics["mean_cer"],
        "top_confusions": ocr_metrics["top_confusions"],
        "avg_psnr_before_after": avg_psnr,
        "avg_ssim_before_after": avg_ssim,
    }

    with open(os.path.join(args.output, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    with open(os.path.join(args.output, "per_sample.csv"), "w", newline="") as f:
        fieldnames = sorted({k for row in per_sample_rows for k in row.keys()})
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(per_sample_rows)

    print(json.dumps(metrics, indent=2))
    print(f"\nDetailed results in {args.output}")


if __name__ == "__main__":
    main()
