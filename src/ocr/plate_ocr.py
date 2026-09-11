from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import List, Optional

import cv2
import numpy as np


# --------------------------------------------------------------------------
# Confusion maps
# --------------------------------------------------------------------------

DIGIT_TO_LETTER_CANDIDATES = {
    "0": ["O", "D", "Q"],
    "1": ["I", "L"],
    "5": ["S"],
    "8": ["B"],
    "2": ["Z"],
    "6": ["G"],
    "4": ["A", "H", "M", "N"],
    "9": ["G", "P"],
}
LETTER_CONFUSIONS = {
    "L": ["D", "I", "E"],
    "D": ["L", "O", "P"],
    "H": ["M", "N", "A"],
    "M": ["A", "H", "N", "W"],
    "N": [ "A", "H", "M"],
    "T": ["I"],
    "I": ["T"],
    "U": ["V"],
    "V": ["U", "Y"],
    "C": ["G", "O"],
    "P": ["R", "F"],
    "R": ["P", "B"],
}
LETTER_TO_DIGIT_CANDIDATES = {
    "O": ["0"],
    "D": ["0"],
    "Q": ["0", "9"],
    "I": ["1"],
    "T": ["1"],
    "L": ["4", "1"],
    "S": ["5"],
    "B": ["8"],
    "Z": ["2"],
    "G": ["6", "9"],
    "A": ["4"],
    "H": ["4"],
}

INDIA_STATE_CODES = {
    "AP", "AR", "AS", "BR", "CG", "GA", "GJ", "HR", "HP", "JK", "JH", "KA",
    "KL", "MP", "MH", "MN", "ML", "MZ", "NL", "OD", "PB", "RJ", "SK", "TN",
    "TS", "TR", "UP", "UK", "WB", "AN", "CH", "DN", "DD", "DL", "LD", "PY", "LA",
}


def _letter_candidates(ch: str) -> List[str]:
    """All plausible LETTER readings for a single OCR'd character."""
    ch = ch.upper()
    cands = [ch] + DIGIT_TO_LETTER_CANDIDATES.get(ch, []) + LETTER_CONFUSIONS.get(ch, [])
    letters = [c for c in dict.fromkeys(cands) if c.isalpha()]
    return letters or [ch]


def _digit_candidates(ch: str) -> List[str]:
    """All plausible DIGIT readings for a single OCR'd character."""
    ch = ch.upper()
    cands = [ch] + LETTER_TO_DIGIT_CANDIDATES.get(ch, [])
    digits = [c for c in dict.fromkeys(cands) if c.isdigit()]
    return digits or [ch]


# --------------------------------------------------------------------------
# Preprocessing helpers (unchanged from your version)
# --------------------------------------------------------------------------


def _ensure_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image.copy()


def _mild_denoise(gray: np.ndarray) -> np.ndarray:
    return cv2.fastNlMeansDenoising(gray, None, h=6, templateWindowSize=3, searchWindowSize=7)


def _enhance_contrast_clahe(gray: np.ndarray, clip_limit: float = 2.0, tile_grid_size: tuple = (4, 4)) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(gray)


def preprocess_for_ocr_natural(
    crop: np.ndarray,
    target_char_height: float = 48.0,
    clip_limit: float = 2.0,
    tile_grid_size: tuple = (4, 4),
) -> np.ndarray:
    gray = _ensure_gray(crop)

    mean_val = float(cv2.mean(gray)[0])
    if mean_val > 200:
        gray = 255 - gray

    h, w = gray.shape
    estimated_char_height = h / 6.0
    if estimated_char_height < target_char_height:
        scale = target_char_height / estimated_char_height
        gray = cv2.resize(gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)

    gray = _mild_denoise(gray)
    gray = _enhance_contrast_clahe(gray, clip_limit=clip_limit, tile_grid_size=tile_grid_size)

    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def preprocess_for_ocr_binary(crop: np.ndarray, invert: bool = False) -> np.ndarray:
    gray = _ensure_gray(crop)

    if invert:
        gray = 255 - gray

    mean_val = float(cv2.mean(gray)[0])
    if mean_val > 180:
        gray = 255 - gray

    denoised = cv2.fastNlMeansDenoising(gray, None, h=8, templateWindowSize=3, searchWindowSize=7)
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]], dtype=np.float32)
    sharpened = cv2.filter2D(denoised, cv2.CV_8U, kernel)
    sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)

    h, w = sharpened.shape
    if h < 64:
        scale = 64.0 / h
        sharpened = cv2.resize(sharpened, (int(w * scale), 64), interpolation=cv2.INTER_CUBIC)

    binary = cv2.adaptiveThreshold(sharpened, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 15)

    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_open, iterations=1)
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1))
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel_close, iterations=1)

    return cleaned


# --------------------------------------------------------------------------
# Segment-based plate correction
#
# [state code: 2 letters] [RTO code: 2 digits] [series: 1-2 letters] [unique number: 4 digits]
#              MH               43                    AJ                    5887
# --------------------------------------------------------------------------


@dataclass
class SegmentResult:
    state_code: str
    rto_code: str
    series: str
    number: str
    state_code_is_valid_rto: bool
    state_code_ambiguous: bool

    @property
    def full_text(self) -> str:
        return self.state_code + self.rto_code + self.series + self.number


def correct_plate_by_segments(text: str) -> Optional[SegmentResult]:
    text = re.sub(r"[^A-Za-z0-9]", "", text).upper()

    for series_len in (2, 1):
        expected_len = 2 + 2 + series_len + 4
        if len(text) != expected_len:
            continue

        seg_state = text[0:2]
        seg_rto = text[2:4]
        seg_series = text[4:4 + series_len]
        seg_number = text[4 + series_len:4 + series_len + 4]

        c0_opts = _letter_candidates(seg_state[0])
        c1_opts = _letter_candidates(seg_state[1])
        valid_pairs = [(a, b) for a in c0_opts for b in c1_opts if (a + b) in INDIA_STATE_CODES]
        if not valid_pairs:
            # Fallback: no candidate-list combination matched directly.
            # Search all real state codes within edit-distance 1 of the raw
            # (uncandidated) state-code string, so we still catch confusions not
            # explicitly listed in DIGIT_TO_LETTER_CANDIDATES/LETTER_CONFUSIONS.
            close_matches = [
                code for code in INDIA_STATE_CODES
                if sum(a != b for a, b in zip(seg_state, code)) <= 1
            ]
            if len(close_matches) == 1:
                valid_pairs = [(close_matches[0][0], close_matches[0][1])]

        if valid_pairs:
            valid_pairs.sort(key=lambda p: (p[0] != seg_state[0]) + (p[1] != seg_state[1]))
            best_pair = valid_pairs[0]
            corrected_state = best_pair[0] + best_pair[1]
            state_valid = True
            state_ambiguous = len(valid_pairs) > 1
        else:
            corrected_state = c0_opts[0] + c1_opts[0]
            state_valid = False
            state_ambiguous = False

        corrected_rto = "".join(_digit_candidates(ch)[0] for ch in seg_rto)
        corrected_series = "".join(_letter_candidates(ch)[0] for ch in seg_series)
        corrected_number = "".join(_digit_candidates(ch)[0] for ch in seg_number)

        return SegmentResult(
            state_code=corrected_state,
            rto_code=corrected_rto,
            series=corrected_series,
            number=corrected_number,
            state_code_is_valid_rto=state_valid,
            state_code_ambiguous=state_ambiguous,
        )

    return None


# --------------------------------------------------------------------------
# OCR result dataclass
#
# `text` / `corrected_text` / `corrected_matches_format` are kept with
# their ORIGINAL names and meanings so existing call sites (e.g.
# streamlit_app.py doing
#   `r.ocr.corrected_text if r.ocr.corrected_matches_format else r.ocr.text`
# ) keep working without any change on your end.
#
# IMPORTANT behavior change from your original script: `corrected_matches_format`
# is now True whenever segment-based correction ran at all (RTO number,
# series, and unique-number segments were type-corrected), not only when
# the 2-letter state code could be confirmed against a real RTO code.
# That's the actual fix for "recovered text not updating" -- previously
# this flag only went True on a full-format match, so any plate with an
# unresolved/ambiguous state code fell back to showing the raw string
# even though the other 8 characters had already been correctly typed.
# Use `segments.state_code_is_valid_rto` (or `needs_review`) if you want
# to separately flag/highlight just the state-code uncertainty in the UI.
# --------------------------------------------------------------------------


@dataclass
class OCRResult:
    text: str
    confidence: float
    matches_known_format: bool
    matched_pattern: Optional[str] = None
    raw_detections: Optional[list] = None

    corrected_text: Optional[str] = None
    corrected_matches_format: Optional[bool] = None
    needs_review: bool = False

    segments: Optional[SegmentResult] = None

    preprocess_variant: str = "none"
    engine_used: str = "unknown"


# --------------------------------------------------------------------------
# PlateOCR class
# --------------------------------------------------------------------------


class PlateOCR:
    def __init__(
        self,
        engine: str = "easyocr",
        languages=None,
        allowlist: str = "",
        min_confidence: float = 0.30,
        format_patterns: Optional[List[str]] = None,
        gpu: bool = True,
        target_char_height: float = 48.0,
        clahe_clip_limit: float = 2.0,
        clahe_tile_grid_size: tuple = (4, 4),
    ):
        self.engine_name = engine
        self.allowlist = allowlist
        self.min_confidence = min_confidence
        self.format_patterns = format_patterns or []
        self._reader = None

        self.target_char_height = target_char_height
        self.clahe_clip_limit = clahe_clip_limit
        self.clahe_tile_grid_size = clahe_tile_grid_size

        if engine == "easyocr":
            import easyocr
            self._reader = easyocr.Reader(languages or ["en"], gpu=gpu)
        elif engine == "tesseract":
            import pytesseract  # noqa: F401
        else:
            raise ValueError(f"Unknown OCR engine: {engine}")

    def _read_easyocr(self, image: np.ndarray, allowlist_override: Optional[str] = None):
        allow = allowlist_override if allowlist_override is not None else (self.allowlist or None)
        results = self._reader.readtext(image, allowlist=allow, paragraph=False, min_size=10)
        if not results:
            return "", 0.0, results
        def box_height(box):
                pts = np.array(box, dtype=np.float32)
                return float(pts[:, 1].max() - pts[:, 1].min())

        heights = [box_height(r[0]) for r in results]
        median_h = np.median(heights)

            # Drop detections much shorter than the median box -- these are
            # almost always stray artifacts (hologram icons, bolts, border
            # lines) rather than genuine plate characters, and including them
            # scrambles read order when sorted by x-position alone.
        filtered = [r for r, h in zip(results, heights) if h >= 0.5 * median_h]
        if not filtered:
            filtered = results  # don't discard everything if all boxes are similarly small

        def centroid_x(box):
            pts = np.array(box, dtype=np.float32)
            return float(pts[:, 0].mean())

        results_sorted = sorted(results, key=lambda r: centroid_x(r[0]))
        text = "".join(r[1] for r in results_sorted).upper().replace(" ", "")
        conf = float(np.mean([r[2] for r in results_sorted]))
        return text, conf, results

    def _read_tesseract(self, image: np.ndarray):
        import pytesseract

        cfg = "--psm 7"
        if self.allowlist:
            cfg += f" -c tessedit_char_whitelist={self.allowlist}"
        data = pytesseract.image_to_data(image, config=cfg, output_type=pytesseract.Output.DICT)
        words, confs = [], []
        for text, conf in zip(data["text"], data["conf"]):
            text = text.strip()
            if text:
                words.append(text)
                confs.append(float(conf) / 100.0 if float(conf) > 0 else 0.0)
        full_text = "".join(words).upper()
        mean_conf = float(np.mean(confs)) if confs else 0.0
        return full_text, mean_conf, data

    def _try_ocr_with_variants(self, image: np.ndarray, preprocess: bool):
        candidates = []

        if self.engine_name == "easyocr":
            variants = [("natural", False)]
            if preprocess:
                variants += [("binary_normal", False), ("binary_inverted", True)]
        else:
            if preprocess:
                variants = [("binary_normal", False), ("binary_inverted", True)]
            else:
                variants = [("raw", False)]

        for name, invert in variants:
            if name == "natural":
                proc = preprocess_for_ocr_natural(
                    image,
                    target_char_height=self.target_char_height,
                    clip_limit=self.clahe_clip_limit,
                    tile_grid_size=self.clahe_tile_grid_size,
                )
            elif name.startswith("binary"):
                proc = preprocess_for_ocr_binary(image, invert=invert)
            else:
                proc = _ensure_gray(image)
                if proc.ndim == 2:
                    proc = cv2.cvtColor(proc, cv2.COLOR_GRAY2BGR)

            if self.engine_name == "easyocr":
                text, conf, raw = self._read_easyocr(proc)
            else:
                text, conf, raw = self._read_tesseract(proc)

            text = re.sub(r"[^A-Za-z0-9]", "", text).upper()
            candidates.append((text, conf, raw, name, self.engine_name))

        def score_candidate(c):
            text, conf, _, _, _ = c
            if len(text) <= 2:
                return (-10.0, 0)

            fmt_ok = (
                any(re.fullmatch(p, text) for p in self.format_patterns)
                if self.format_patterns else False
            )

            length = len(text)
            length_score = 0.0
            if 8 <= length <= 12:
                length_score += 0.3
            if 9 <= length <= 11:
                length_score += 0.2

            has_letter = any(ch.isalpha() for ch in text)
            has_digit = any(ch.isdigit() for ch in text)
            mixed_score = 0.2 if (has_letter and has_digit) else 0.0

            total_score = conf + (0.4 if fmt_ok else 0.0) + length_score + mixed_score
            return (total_score, length)

        candidates.sort(key=score_candidate, reverse=True)
        return candidates[0]

    def read(self, image: np.ndarray, preprocess: bool = True, correct: bool = True) -> OCRResult:
        text, conf, raw, prep_variant, engine_used = self._try_ocr_with_variants(image, preprocess)

        matched_pattern = None
        for pattern in self.format_patterns:
            if re.fullmatch(pattern, text):
                matched_pattern = pattern
                break

        segments: Optional[SegmentResult] = None
        corrected_text: Optional[str] = None
        corrected_matches_format: Optional[bool] = None

        if correct and text:
            segments = correct_plate_by_segments(text)
            if segments is not None:
                corrected_text = segments.full_text
    
                # Guard against false-positive "matches format" on garbled input:
                # count how many characters actually changed during correction.
                if corrected_text is None:
                    num_changed = len(text)
                elif len(text) == len(corrected_text):
                    num_changed = sum(a != b for a, b in zip(text, corrected_text))
                else:
                    num_changed = len(text)
    
                correction_ratio = num_changed / max(len(text), 1)
    
                corrected_matches_format = segments.state_code_is_valid_rto and correction_ratio <= 0.3
    
                if matched_pattern is None and corrected_matches_format:
                    matched_pattern = "indian_plate_segments"

        format_ok = (matched_pattern is not None) or bool(corrected_matches_format)
        needs_review = (
            (conf < self.min_confidence)
            or not format_ok
            or (segments is not None and not segments.state_code_is_valid_rto)
            or (segments is not None and segments.state_code_ambiguous)
        )

        return OCRResult(
            text=text,
            confidence=conf,
            matches_known_format=matched_pattern is not None and matched_pattern != "indian_plate_segments",
            matched_pattern=matched_pattern,
            raw_detections=raw,
            corrected_text=corrected_text,
            corrected_matches_format=corrected_matches_format,
            needs_review=needs_review,
            segments=segments,
            preprocess_variant=prep_variant,
            engine_used=engine_used,
        )


# --------------------------------------------------------------------------
# Temporal voting across video frames
# --------------------------------------------------------------------------


def vote_across_frames(results: List[OCRResult], use_corrected: bool = True) -> Optional[str]:
    texts = [
        (r.corrected_text if use_corrected and r.corrected_text else r.text)
        for r in results
        if (r.corrected_text if use_corrected and r.corrected_text else r.text)
    ]
    if not texts:
        return None

    length = Counter(len(t) for t in texts).most_common(1)[0][0]
    same_len = [t for t in texts if len(t) == length]
    if not same_len:
        return None

    voted = []
    for i in range(length):
        counts = Counter(t[i] for t in same_len)
        voted.append(counts.most_common(1)[0][0])
    return "".join(voted)