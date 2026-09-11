"""
OCR accuracy metrics against ground-truth plate strings.

Design choice
-------------
This is the primary success metric for the assignment (the task's actual
goal is "recover readable plate text", not "produce a pretty image"), so we
report several complementary views:

  - Exact match accuracy: strict, most relevant for real forensic/legal use
    where a single wrong character invalidates a lookup.
  - Character Error Rate (CER): Levenshtein edit distance / reference
    length -- more informative than exact-match alone, since it shows
    "how close" a wrong prediction was (useful for triaging which
    degradation types are hardest, per the assignment's failure-case
    analysis requirement).
  - Per-character confusion pairs: surfaces systematic OCR errors (e.g.
    0/O, 8/B, 1/I) that are common with plate fonts, which is useful, 
    actionable qualitative-analysis material for the report.
"""

from __future__ import annotations

from collections import Counter
from typing import List, Tuple

try:
    import Levenshtein
except ImportError:  # pragma: no cover - exercised only when the optional
    # dependency isn't installed; keeps evaluation usable in minimal envs.
    class _LevenshteinFallback:
        @staticmethod
        def distance(a: str, b: str) -> int:
            # Standard O(len(a)*len(b)) edit-distance DP fallback.
            if a == b:
                return 0
            m, n = len(a), len(b)
            dp = list(range(n + 1))
            for i in range(1, m + 1):
                prev, dp[0] = dp[0], i
                for j in range(1, n + 1):
                    cur = dp[j]
                    cost = 0 if a[i - 1] == b[j - 1] else 1
                    dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + cost)
                    prev = cur
            return dp[n]

        @staticmethod
        def editops(a: str, b: str) -> List[tuple]:
            # Minimal editops fallback (replace-only alignment via DP
            # backtrace); sufficient for confusion-pair reporting even
            # though it won't distinguish insert/delete as precisely as
            # python-Levenshtein for length-mismatched strings.
            m, n = len(a), len(b)
            dp = [[0] * (n + 1) for _ in range(m + 1)]
            for i in range(m + 1):
                dp[i][0] = i
            for j in range(n + 1):
                dp[0][j] = j
            for i in range(1, m + 1):
                for j in range(1, n + 1):
                    cost = 0 if a[i - 1] == b[j - 1] else 1
                    dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
            ops, i, j = [], m, n
            while i > 0 and j > 0:
                if a[i - 1] == b[j - 1]:
                    i, j = i - 1, j - 1
                elif dp[i][j] == dp[i - 1][j - 1] + 1:
                    ops.append(("replace", i - 1, j - 1))
                    i, j = i - 1, j - 1
                elif dp[i][j] == dp[i - 1][j] + 1:
                    ops.append(("delete", i - 1, j))
                    i -= 1
                else:
                    ops.append(("insert", i, j - 1))
                    j -= 1
            return ops

    Levenshtein = _LevenshteinFallback()


def exact_match(pred: str, gt: str) -> bool:
    return pred.strip().upper() == gt.strip().upper()


def character_error_rate(pred: str, gt: str) -> float:
    gt = gt.strip().upper()
    pred = pred.strip().upper()
    if len(gt) == 0:
        return 1.0 if len(pred) > 0 else 0.0
    return Levenshtein.distance(pred, gt) / len(gt)


def confusion_pairs(pred: str, gt: str) -> List[Tuple[str, str]]:
    """Character-level substitutions implied by the edit-distance alignment."""
    ops = Levenshtein.editops(pred.strip().upper(), gt.strip().upper())
    pairs = []
    for op, i, j in ops:
        if op == "replace":
            pairs.append((pred[i], gt[j]))
    return pairs


def evaluate_dataset(predictions: List[str], ground_truths: List[str]) -> dict:
    assert len(predictions) == len(ground_truths)
    n = len(predictions)
    exact = [exact_match(p, g) for p, g in zip(predictions, ground_truths)]
    cers = [character_error_rate(p, g) for p, g in zip(predictions, ground_truths)]

    all_confusions = Counter()
    for p, g in zip(predictions, ground_truths):
        all_confusions.update(confusion_pairs(p, g))

    return {
        "n": n,
        "exact_match_accuracy": sum(exact) / n if n else 0.0,
        "mean_cer": sum(cers) / n if n else 0.0,
        "top_confusions": all_confusions.most_common(10),
        "per_sample": [
            {"pred": p, "gt": g, "exact": e, "cer": c}
            for p, g, e, c in zip(predictions, ground_truths, exact, cers)
        ],
    }
