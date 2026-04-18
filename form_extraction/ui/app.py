"""Streamlit UI for Form 283 extraction.

Imports only from `form_extraction.core`. The UI layer does three things:
  * surfaces the extracted JSON (canonical + Hebrew-labelled),
  * shows the raw OCR next to it for human-in-the-loop accuracy review,
  * surfaces format issues from the validator.

Run with:
    streamlit run form_extraction/ui/app.py
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import streamlit as st
from openai import APIConnectionError, APIStatusError, AuthenticationError, RateLimitError
from pydantic import ValidationError

from form_extraction.core.pipeline import run
from form_extraction.core.schemas import to_hebrew_keys

# Configure logging once per process. Streamlit reruns the script but logging
# handlers survive, so basicConfig is a no-op on subsequent runs.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
log = logging.getLogger("form_extraction.ui")

st.set_page_config(page_title="Form 283 Extractor", layout="wide")
st.title("Bituach Leumi – Form 283 Extractor")
st.caption(
    "Upload a filled form (PDF/JPG/PNG). "
    "Azure Document Intelligence runs OCR, GPT-4o returns a JSON payload, "
    "and a small rule check flags obvious format errors."
)

SAMPLE_DIR = Path(__file__).resolve().parents[2] / "phase1_data"


def _find_sample() -> Path | None:
    if not SAMPLE_DIR.is_dir():
        return None
    for pdf in sorted(SAMPLE_DIR.glob("283_ex*.pdf")):
        return pdf
    return None


def _describe_error(exc: BaseException) -> str:
    """Turn SDK / pipeline errors into a short actionable message."""
    if isinstance(exc, AuthenticationError):
        return (
            "Azure OpenAI rejected the API key. "
            "Check AZURE_OPENAI_KEY and AZURE_OPENAI_ENDPOINT in your .env."
        )
    if isinstance(exc, RateLimitError):
        return "Azure OpenAI rate-limited the request. Wait a moment and try again."
    if isinstance(exc, APIConnectionError):
        return "Could not reach Azure OpenAI. Check your network and endpoint URL."
    if isinstance(exc, APIStatusError):
        return f"Azure OpenAI returned HTTP {exc.status_code}. Try again or check the service status."
    if isinstance(exc, ValidationError):
        return "The model returned JSON that does not match the expected schema."
    if isinstance(exc, RuntimeError):
        return str(exc)
    return f"Unexpected error: {exc.__class__.__name__}"


col_upload, col_options = st.columns([3, 1])
with col_upload:
    uploaded = st.file_uploader("Upload a form", type=["pdf", "jpg", "jpeg", "png"])
with col_options:
    hebrew_keys = st.toggle("Show Hebrew field labels", value=False)

sample_path = _find_sample()
file_bytes: bytes | None = None
file_label: str | None = None

if uploaded is not None:
    file_bytes = uploaded.getvalue()
    file_label = uploaded.name
elif sample_path is not None:
    if st.button(f"Load sample: {sample_path.name}"):
        file_bytes = sample_path.read_bytes()
        file_label = sample_path.name

if file_bytes is None:
    st.info("Upload a form to get started." + ("" if sample_path else " (No bundled sample available.)"))
    st.stop()

if st.button("Extract fields", type="primary"):
    log.info("ui.extract label=%s bytes=%d", file_label, len(file_bytes))
    try:
        with st.spinner("Running OCR..."):
            # The pipeline runs end-to-end; this spinner covers the whole trip.
            result = run(file_bytes)
    except Exception as exc:
        log.exception("ui.extract_failed label=%s", file_label)
        st.error(_describe_error(exc))
        st.stop()
    st.session_state["result"] = {
        "form": result.form.model_dump(),
        "report": result.report.to_dict(),
        "ocr_text": result.ocr_text,
    }

result = st.session_state.get("result")
if not result:
    st.stop()

report = result["report"]
col1, col2, col3 = st.columns(3)
col1.metric("Completeness", f"{report['completeness'] * 100:.1f}%")
col2.metric("Filled", f"{report['filled']} / {report['total']}")
col3.metric("Issues", len(report["issues"]))

tab_json, tab_ocr, tab_issues = st.tabs(["Extracted JSON", "OCR text", "Validation"])

with tab_json:
    shown = to_hebrew_keys(result["form"]) if hebrew_keys else result["form"]
    st.json(shown, expanded=True)
    dl_col1, dl_col2 = st.columns(2)
    dl_col1.download_button(
        "Download JSON (English keys)",
        data=json.dumps(result["form"], ensure_ascii=False, indent=2).encode("utf-8"),
        file_name="form283.json",
        mime="application/json",
    )
    dl_col2.download_button(
        "Download JSON (Hebrew keys)",
        data=json.dumps(to_hebrew_keys(result["form"]), ensure_ascii=False, indent=2).encode("utf-8"),
        file_name="form283_he.json",
        mime="application/json",
    )

with tab_ocr:
    st.caption("Raw OCR output — compare side-by-side with the extracted JSON to verify accuracy.")
    st.code(result["ocr_text"], language="markdown")

with tab_issues:
    if report["issues"]:
        st.table(report["issues"])
    else:
        st.success("No format issues detected.")
