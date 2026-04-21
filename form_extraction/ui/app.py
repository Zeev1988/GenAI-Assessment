"""Streamlit UI for Form 283 extraction."""

from __future__ import annotations

import json
from pathlib import Path

import streamlit as st
from openai import APIConnectionError, APIStatusError, AuthenticationError, RateLimitError
from pydantic import ValidationError

from common import get_logger
from form_extraction.core.pipeline import run
from form_extraction.core.schemas import to_hebrew_keys

log = get_logger("form_extraction.ui")

st.set_page_config(page_title="Form 283 Extractor", layout="wide")
st.title("Bituach Leumi – Form 283 Extractor")
st.caption(
    "Upload a filled form (PDF/JPG/PNG). Azure Document Intelligence runs OCR, "
    "GPT-4o returns JSON, and a small rule check flags format errors."
)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SAMPLE_DIRS = (
    _PROJECT_ROOT / "test_data",
    _PROJECT_ROOT / "tests" / "test_data" / "test_data",
    _PROJECT_ROOT / "tests" / "test_data",
)


def _find_sample() -> Path | None:
    for sample_dir in _SAMPLE_DIRS:
        if not sample_dir.is_dir():
            continue
        for pdf in sorted(sample_dir.glob("283_ex*.pdf")):
            return pdf
    return None


def _describe_error(exc: BaseException) -> str:
    if isinstance(exc, AuthenticationError):
        return "Azure OpenAI rejected the API key. Check AZURE_OPENAI_KEY / AZURE_OPENAI_ENDPOINT."
    if isinstance(exc, RateLimitError):
        return "Azure OpenAI rate-limited the request. Wait a moment and try again."
    if isinstance(exc, APIConnectionError):
        return "Could not reach Azure OpenAI. Check your network and endpoint URL."
    if isinstance(exc, APIStatusError):
        return f"Azure OpenAI returned HTTP {exc.status_code}."
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
    st.info("Upload a form to get started.")
    st.stop()

if st.button("Extract fields", type="primary"):
    log.info("ui.extract label=%s bytes=%d", file_label, len(file_bytes))
    try:
        with st.spinner("Running OCR..."):
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
    st.caption("Raw OCR output — compare with the extracted JSON.")
    st.code(result["ocr_text"], language="markdown")

with tab_issues:
    if report["issues"]:
        st.table(report["issues"])
    else:
        st.success("No format issues detected.")
