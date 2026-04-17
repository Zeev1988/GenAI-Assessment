"""Streamlit UI for uploading a Form 283 and viewing extracted JSON."""

from __future__ import annotations

import json
import uuid
from typing import Any

import pandas as pd
import streamlit as st
from common.config import get_settings
from common.errors import AppError
from common.logging_config import configure_logging, get_logger

from form_extraction.backend.pipeline import run_pipeline_sync
from form_extraction.backend.schemas import to_hebrew_keys

configure_logging(get_settings().app_log_level)
_log = get_logger("ui")


st.set_page_config(
    page_title="Bituach Leumi - Form 283 Extractor",
    page_icon=":page_facing_up:",
    layout="wide",
)


def _init_state() -> None:
    st.session_state.setdefault("result", None)
    st.session_state.setdefault("error", None)
    st.session_state.setdefault("correlation_id", "")


def _render_sidebar() -> tuple[Any, bool, str, bool]:
    with st.sidebar:
        st.header("Upload")
        uploaded = st.file_uploader(
            "PDF, JPG, or PNG",
            type=["pdf", "jpg", "jpeg", "png"],
            accept_multiple_files=False,
        )
        st.header("Options")
        hebrew_keys = st.toggle(
            "Show Hebrew keys",
            value=False,
            help="Re-label the output JSON with the Hebrew field names from the assignment.",
        )
        language_hint = st.selectbox(
            "Language hint",
            options=["auto", "he", "en", "mixed"],
            index=0,
            help="Only used as a hint; the OCR auto-detects language.",
        )
        run_clicked = st.button(
            "Extract fields",
            type="primary",
            use_container_width=True,
            disabled=uploaded is None,
        )
    return uploaded, hebrew_keys, language_hint, run_clicked


def _severity_icon(severity: str) -> str:
    return {"error": "X", "warning": "!", "info": "i"}.get(severity, "*")


def _render_report(report: dict[str, Any]) -> None:
    completeness = report["completeness"]
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Completeness", f"{completeness * 100:.1f}%")
    col2.metric("Filled fields", f"{report['filled_fields']} / {report['total_fields']}")
    col3.metric("Errors", report["error_count"])
    col4.metric("Warnings", report["warning_count"])
    st.progress(min(max(completeness, 0.0), 1.0))

    issues = report.get("issues") or []
    if issues:
        df = pd.DataFrame(
            [
                {
                    "severity": _severity_icon(i["severity"]) + " " + i["severity"],
                    "field": i["field"],
                    "message": i["message"],
                }
                for i in issues
            ]
        )
        st.dataframe(df, hide_index=True, use_container_width=True)
    else:
        st.success("No validation issues detected.")

    if report.get("judge_score") is not None:
        st.caption(f"LLM-judge faithfulness score: {report['judge_score']}/100")
        for comment in report.get("judge_comments", []):
            st.caption(f"- {comment}")


def _render_timings(result: dict[str, Any]) -> None:
    timings = result.get("stage_timings_ms", {})
    from_cache = result.get("from_cache", {})
    rows = [
        {
            "stage": stage,
            "ms": round(ms, 1),
            "from_cache": from_cache.get(stage.replace("_ms", ""), False),
        }
        for stage, ms in timings.items()
    ]
    if rows:
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)


def main() -> None:
    _init_state()
    st.title("Bituach Leumi - Form 283 Extractor")
    st.caption(
        "Upload a filled Bituach Leumi form (PDF/JPG/PNG). "
        "Azure Document Intelligence runs OCR, GPT-4o returns a validated JSON."
    )

    uploaded, hebrew_keys, language_hint, run_clicked = _render_sidebar()

    if run_clicked and uploaded is not None:
        st.session_state["result"] = None
        st.session_state["error"] = None
        cid = uuid.uuid4().hex
        st.session_state["correlation_id"] = cid

        with st.spinner("Running OCR, extracting fields, validating..."):
            try:
                data = uploaded.getvalue()
                result = run_pipeline_sync(
                    data,
                    filename=uploaded.name,
                    correlation_id=cid,
                )
                st.session_state["result"] = result.to_dict()
            except AppError as exc:
                st.session_state["error"] = {
                    "code": exc.code,
                    "message": exc.message,
                    "details": exc.details,
                }
            except Exception as exc:
                _log.exception("ui.unexpected_error", error=str(exc))
                st.session_state["error"] = {
                    "code": "unexpected_error",
                    "message": str(exc),
                    "details": {},
                }

    if st.session_state["error"]:
        err = st.session_state["error"]
        st.error(f"{err['code']}: {err['message']}")
        if err.get("details"):
            with st.expander("Error details"):
                st.json(err["details"])
        return

    result = st.session_state["result"]
    if not result:
        st.info("Upload a form and click **Extract fields** to get started.")
        return

    st.caption(
        f"Correlation ID: `{result.get('correlation_id', '')}` - "
        f"fingerprint: `{result.get('fingerprint', '')[:12]}...` - "
        f"mime: `{result.get('mime', '')}` - "
        f"language: `{result['ocr']['language_hint']}`"
    )

    tab_json, tab_report, tab_ocr, tab_meta = st.tabs(
        ["Extracted JSON", "Validation", "OCR preview", "Timings"]
    )

    extracted = result["extracted"]
    displayed = to_hebrew_keys(extracted) if hebrew_keys else extracted

    with tab_json:
        st.json(displayed, expanded=True)
        st.download_button(
            "Download JSON",
            data=json.dumps(displayed, ensure_ascii=False, indent=2).encode("utf-8"),
            file_name="form283_extracted.json",
            mime="application/json",
        )

    with tab_report:
        _render_report(result["validation"])

    with tab_ocr:
        st.caption(
            f"{result['ocr']['page_count']} page(s), "
            f"{result['ocr']['char_count']} characters, "
            f"tables: {result['ocr']['has_tables']}"
        )
        with st.expander("Show raw OCR content (may contain PII)"):
            st.text("Hidden by default to respect privacy. Enable via the app logs.")

    with tab_meta:
        _render_timings(result)

    _ = language_hint


if __name__ == "__main__":
    main()
