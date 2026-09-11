"""
Interactive demo app for the forensic license-plate pipeline.

Run with:
    streamlit run app/streamlit_app.py

Lets a reviewer (or interviewer) drop in a new image/video, see the plate
detection, walk through every enhancement stage side-by-side, and see the
final OCR text + confidence — this directly supports the assignment's
requirement to "keep your environment ready to accept new inputs."
"""

import os
import sys
import tempfile

import cv2
import numpy as np
import streamlit as st

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.pipeline import ForensicPlatePipeline  # noqa: E402


st.set_page_config(page_title="Forensic Plate Recovery", layout="wide")

CONFIG_PATH = "configs/config.yaml"
FRAME_STRIDE = 2


@st.cache_resource
def load_pipeline(config_path: str):
    return ForensicPlatePipeline(config_path=config_path)


def bgr_to_rgb(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img.ndim == 3 else img


def render_plate_result(r, index: int):
    st.markdown(f"#### Plate {index + 1}")

    stage_cols = st.columns(len(r.stages))
    for col, (stage_name, img) in zip(stage_cols, r.stages.items()):
        with col:
            st.image(bgr_to_rgb(img), caption=stage_name, use_container_width=True)

    display_text = r.ocr.corrected_text if r.ocr.corrected_matches_format else r.ocr.text
    text_col, conf_col, fmt_col = st.columns(3)
    text_col.metric("Recovered text", display_text or "—")
    conf_col.metric("OCR confidence", f"{r.ocr.confidence:.2f}")
    fmt_col.metric(
        "Matches known format",
        "✅" if (r.ocr.matches_known_format or r.ocr.corrected_matches_format) else "❌",
    )

    if r.ocr.corrected_text and r.ocr.corrected_text != r.ocr.text:
        st.caption(
            f"Raw OCR reading: `{r.ocr.text}` → corrected to `{r.ocr.corrected_text}` "
            f"to match a known plate format."
        )

    st.caption(
        f"Detection confidence: {r.detection.confidence:.2f} · "
        f"BBox: {r.detection.bbox} · Processing time: {r.timing_sec:.2f}s"
    )
    st.markdown("---")


def main():
    st.title("🔍 Forensic License Plate Enhancement & Recovery")
    st.caption(
        "Detect → correct → denoise → deblur → super-resolve → contrast-enhance → OCR"
    )

    pipeline = load_pipeline(CONFIG_PATH)

    uploaded = st.file_uploader(
        "Upload an image or video with a degraded license plate",
        type=["jpg", "jpeg", "png", "bmp", "mp4", "avi", "mov"],
    )

    if uploaded is None:
        st.info("Upload a file to run the pipeline.")
        return

    suffix = os.path.splitext(uploaded.name)[1].lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded.read())
        tmp_path = tmp.name

    is_video = suffix in {".mp4", ".avi", ".mov"}

    input_col, output_col = st.columns(2)

    with input_col:
        st.subheader("Input")
        if is_video:
            st.video(tmp_path)
        else:
            image = cv2.imread(tmp_path)
            st.image(bgr_to_rgb(image), use_container_width=True)

    with output_col:
        st.subheader("Results")
        with st.spinner("Running pipeline..."):
            if is_video:
                results = pipeline.process_video(tmp_path, frame_stride=FRAME_STRIDE)
            else:
                results = pipeline.process_image(image)

        if not results:
            st.warning(
                "No license plate detected. Try a different frame/crop, or "
                "check that the detector weights are appropriate for this footage."
            )
            return

        st.success(f"Detected {len(results)} plate(s).")
        for i, r in enumerate(results):
            render_plate_result(r, i)


if __name__ == "__main__":
    main()