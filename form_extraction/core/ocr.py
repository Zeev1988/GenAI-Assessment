"""Run Azure Document Intelligence (prebuilt-layout) on an uploaded file."""

from __future__ import annotations

import logging
import re
import time

from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import AnalyzeDocumentRequest, DocumentContentFormat
from azure.core.credentials import AzureKeyCredential

from form_extraction.core.config import Settings, get_settings

log = logging.getLogger("form_extraction.ocr")


def run_ocr(data: bytes, settings: Settings | None = None) -> str:
    """Return augmented OCR content produced by prebuilt-layout.

    The raw Markdown is prepended with a SELECTED CHECKBOXES block derived
    from Azure DI's own selection-mark detections (which carry spatial
    bounding-box data). This is necessary because the Markdown text
    representation of RTL Hebrew forms often places ☒/☐ symbols next to the
    wrong labels; the spatial data is always correct regardless of reading
    direction.
    """
    s = settings or get_settings()
    if not s.azure_doc_intelligence_endpoint or not s.azure_doc_intelligence_key.get_secret_value():
        raise RuntimeError(
            "Azure Document Intelligence is not configured. "
            "Set AZURE_DOC_INTELLIGENCE_ENDPOINT and AZURE_DOC_INTELLIGENCE_KEY."
        )

    log.info("ocr.start bytes=%d", len(data))
    t0 = time.perf_counter()

    client = DocumentIntelligenceClient(
        endpoint=s.azure_doc_intelligence_endpoint,
        credential=AzureKeyCredential(s.azure_doc_intelligence_key.get_secret_value()),
    )
    with client:
        poller = client.begin_analyze_document(
            model_id="prebuilt-layout",
            body=AnalyzeDocumentRequest(bytes_source=data),
            locale="he",  # Informs DI of RTL reading direction; may improve checkbox ordering in Markdown
            output_content_format=DocumentContentFormat.MARKDOWN,
        )
        result = poller.result()

    content = (result.content or "").strip()
    elapsed_ms = int((time.perf_counter() - t0) * 1000)
    if not content:
        log.warning("ocr.empty elapsed_ms=%d", elapsed_ms)
        raise RuntimeError("OCR returned empty content; the document may be blank or unreadable.")

    # Azure DI renders individual digit-box fields (dates, IDs) as spaced tokens
    # with brackets and pipes: "[3 ] 0 0 | 6 1 9 9 9" instead of "30061999".
    # Collapse these before the LLM sees the text.
    content, n_collapsed = _collapse_spaced_digits(content)
    if n_collapsed:
        log.info("ocr.collapsed_digit_sequences count=%d", n_collapsed)

    # Normalise the ID number.  When Hebrew text (e.g. "עי") appears before the
    # digit run, Azure DI's bidi algorithm reverses the digit order.  We detect
    # this in Python (unambiguously) and prepend a NORMALIZED_ID tag so the LLM
    # does not have to decide which case it is seeing.
    normalized_id = _extract_normalized_id(content)
    if normalized_id:
        content = f"NORMALIZED_ID: {normalized_id}\n" + content
        log.info("ocr.normalized_id value=%r", normalized_id)

    # Prepend spatial checkbox data so the LLM doesn't have to rely on the
    # potentially mis-ordered ☒/☐ symbols in the Markdown text.
    selected_labels = _extract_selected_labels(result)
    if selected_labels:
        header_lines = [
            "SELECTED CHECKBOXES (authoritative — derived from spatial bounding-box data,",
            "not from the ☒/☐ symbols below which may be misplaced due to RTL processing):",
        ]
        for label in selected_labels:
            header_lines.append(f"  • {label}")
        header_lines += [
            "",
            "For every checkbox field use ONLY the labels listed above.",
            "Ignore ☒/☐/:selected: markers in the OCR text that follows.",
            "---",
            "",
        ]
        content = "\n".join(header_lines) + content
        log.info("ocr.selected_marks count=%d", len(selected_labels))

    # Resolve healthFundMember in Python from the authoritative selected_labels
    # list and emit a NORMALIZED_HEALTH_FUND tag so the LLM never has to infer
    # it from body text.  An empty tag means "no fund checkbox was marked".
    health_fund = _resolve_health_fund(selected_labels)
    content = f"NORMALIZED_HEALTH_FUND: {health_fund}\n" + content
    log.info("ocr.normalized_health_fund value=%r", health_fund)

    log.info("ocr.done chars=%d elapsed_ms=%d", len(content), elapsed_ms)
    return content


# ---------------------------------------------------------------------------
# OCR text normalisation
# ---------------------------------------------------------------------------

# Matches exactly 8 digits separated by any mix of spaces, brackets, and
# pipes — the artefact produced when Azure DI scans a form whose date/ID
# fields are printed as individual digit boxes.
# Examples it must handle:
#   "[3 ] 0 0 | 6 1 9 9 9"  →  "30061999"
#   "[2| 0 0 5 1 9 9 9"     →  "20051999"
#   "[1 ] 4 0 4 1 | 9 9 9"  →  "14041999"
#   "[0 3 0 3 1 9 7 4"      →  "03031974"
_SPACED_DIGIT_RE = re.compile(
    r"\[?"           # optional leading bracket
    r"\d"            # digit 1
    r"[\s\[\]|]*"
    r"\d"            # digit 2
    r"[\s\[\]|]*"
    r"\d"            # digit 3
    r"[\s\[\]|]*"
    r"\d"            # digit 4
    r"[\s\[\]|]*"
    r"\d"            # digit 5
    r"[\s\[\]|]*"
    r"\d"            # digit 6
    r"[\s\[\]|]*"
    r"\d"            # digit 7
    r"[\s\[\]|]*"
    r"\d"            # digit 8
)


def _collapse_spaced_digits(text: str) -> tuple[str, int]:
    """Replace spaced 8-digit sequences with their compact form.

    Returns the cleaned text and the number of substitutions made.
    Only collapses when exactly 8 digits are found in the match (guards
    against accidentally merging unrelated short digit tokens).
    """
    count = 0

    def _replace(m: re.Match) -> str:
        nonlocal count
        digits = re.sub(r"\D", "", m.group(0))
        if len(digits) == 8:
            count += 1
            log.debug("ocr.collapse %r → %r", m.group(0), digits)
            return digits
        return m.group(0)

    cleaned = _SPACED_DIGIT_RE.sub(_replace, text)
    return cleaned, count


# ---------------------------------------------------------------------------
# Checkbox normalisation helpers
# ---------------------------------------------------------------------------

_FUND_NAMES: frozenset[str] = frozenset({"כללית", "מכבי", "מאוחדת", "לאומית"})


def _resolve_health_fund(selected_labels: list[str]) -> str:
    """Return the fund name from the selected labels, or "" if none is selected.

    The spatial selection-mark resolver already maps every checked box to its
    canonical Hebrew label.  We scan for a fund name here in Python so the LLM
    receives a definitive NORMALIZED_HEALTH_FUND value and cannot infer the fund
    from printed labels elsewhere on the form.
    """
    for label in selected_labels:
        if label in _FUND_NAMES:
            return label
    return ""


# ---------------------------------------------------------------------------
# ID number normalisation
# ---------------------------------------------------------------------------

# Matches the "ת. ז." label (with variable spacing/punctuation) followed by
# anything on the same line, then captures up to two following lines that
# contain the raw digit run (possibly prefixed by Hebrew noise).
_ID_LABEL_RE = re.compile(
    r"ת[. ]*ז[., ]*[^\n]*\n",   # "ת. ז." header line
    re.UNICODE,
)

# Matches one or more Hebrew alphabet characters (including geresh/gershayim).
_HEBREW_RE = re.compile(r"[\u0590-\u05FF\uFB1D-\uFB4F\u05F3\u05F4\"״׳]", re.UNICODE)

# A line that contains at least one digit (the ID digit run).
_DIGIT_LINE_RE = re.compile(r"[^\n]*\d[^\n]*")


def _extract_normalized_id(text: str) -> str:
    """Return the ID digit string extracted from the OCR, or "" if not found.

    Scans up to 4 lines after the "ת. ז." label for the first line that
    contains digits (skipping Hebrew-only interstitial rows like "ס״ב").

    Detects whether Hebrew characters precede the digit run:
    - CASE A (Hebrew prefix present): bidi reversed the digit order → reverse.
    - CASE B (digits only): correct order as-is.

    The result is returned verbatim — no length trimming.  If the form has a
    10-digit ID the validator will surface the error; we must not silently drop
    a digit that was genuinely written on the form.
    """
    m = _ID_LABEL_RE.search(text)
    if not m:
        return ""

    # Scan the next few lines after the label for the first that contains digits
    rest = text[m.end():]
    raw_line = ""
    for line in rest.splitlines()[:4]:
        if re.search(r"\d", line):
            raw_line = line
            break

    if not raw_line:
        return ""

    digits = re.sub(r"\D", "", raw_line)
    if not digits:
        return ""

    has_hebrew = bool(_HEBREW_RE.search(raw_line))
    log.info("ocr.id_raw line=%r has_hebrew=%s digits=%r", raw_line.strip(), has_hebrew, digits)

    if has_hebrew:
        # Case A: reverse because bidi flipped the digit sequence
        digits = digits[::-1]
        log.info("ocr.id_reversed digits=%r", digits)

    return digits  # validator will flag if length != 9


# ---------------------------------------------------------------------------
# Spatial helpers
# ---------------------------------------------------------------------------

def _poly_center(polygon: list[float]) -> tuple[float, float]:
    """Return (cx, cy) of a flat [x0, y0, x1, y1, …] polygon list."""
    xs = polygon[0::2]
    ys = polygon[1::2]
    return sum(xs) / len(xs), sum(ys) / len(ys)


def _poly_height(polygon: list[float]) -> float:
    ys = polygon[1::2]
    return max(ys) - min(ys) if ys else 1.0


def _resolve_label(nearby: list[tuple[float, str]]) -> str:
    """Map the nearest-word candidates to a canonical Form 283 checkbox label.

    The raw nearby-word list is noisy: section-header words (e.g. 'מקום',
    'התאונה:', 'למילוי') can appear closer to the mark than the actual option
    label because of how the form is laid out.  Rather than passing that noise
    to the LLM, we resolve to the exact Hebrew label string here in Python.

    Priority order:
    1. Single-word options — whichever known option word appears first in the
       distance-sorted list (within 2 inches) wins.
    2. Multi-word membership options — detected by the presence of 'הנפגע'
       combined with (or without) 'אינו'.
    3. Multi-word accident-location options — detected by key words.
    4. Fallback — the 3 nearest raw words joined by a space.
    """
    word_set = {w for _, w in nearby}

    # --- 1. Single-word checkbox options (exact match, closest wins) ---------
    _SINGLE = {
        "זכר", "נקבה",                                   # gender
        "כללית", "מכבי", "מאוחדת", "לאומית",             # health fund
        "במפעל", "אחר",                                   # accident location
    }
    for dist, word in nearby:  # already sorted by distance
        if dist > 0.5:
            break  # nothing closer than 0.5 inches qualifies as the label for this mark
        if word in _SINGLE:
            return word

    # --- 2. Membership-status pair (both start with 'הנפגע') -----------------
    if "הנפגע" in word_set:
        if "אינו" in word_set:
            return "הנפגע אינו חבר בקופת חולים"
        return "הנפגע חבר בקופת חולים"

    # --- 3. Multi-word accident-location options -----------------------------
    if "ת." in word_set and "דרכים" in word_set:
        if "בדרך" in word_set or "לעבודה" in word_set or "מהעבודה" in word_set:
            return "ת. דרכים בדרך לעבודה/מהעבודה"
        return "ת. דרכים בעבודה"
    if "בדרך" in word_set:
        if "לעבודה" in word_set:
            return "בדרך לעבודה"
        if "מהעבודה" in word_set:
            return "בדרך מהעבודה"
    if "מחוץ" in word_set:
        return "מחוץ למפעל"

    # --- 4. Fallback ---------------------------------------------------------
    return " ".join(w for _, w in nearby[:3])


def _extract_selected_labels(result, words_per_mark: int = 5) -> list[str]:
    """Return text labels for every 'selected' selection mark.

    Azure DI records each selection mark with a polygon (bounding box) that is
    independent of the text reading order.  We find words whose vertical centre
    falls within ±1.5× the mark's own height of the mark's vertical centre —
    i.e. words on the same visual row — then take the few closest ones as the
    label.  This is reliable for both LTR and RTL layouts.
    """
    selected_labels: list[str] = []

    for page_idx, page in enumerate(result.pages or []):
        marks = page.selection_marks or []
        words = page.words or []

        # Log ALL marks (selected and unselected) so we can audit spatial assignments.
        log.info(
            "ocr.spatial_debug page=%d total_marks=%d total_words=%d",
            page_idx + 1, len(marks), len(words),
        )
        for i, mark in enumerate(marks):
            state = getattr(mark, "state", "") or ""
            polygon = getattr(mark, "polygon", None)
            if polygon and len(polygon) >= 4:
                mcx, mcy = _poly_center(polygon)
                mh = _poly_height(polygon)
                log.info(
                    "ocr.spatial_debug   mark[%d] state=%s center=(%.3f, %.3f) height=%.3f",
                    i, state, mcx, mcy, mh,
                )
            else:
                log.info("ocr.spatial_debug   mark[%d] state=%s polygon=MISSING", i, state)

        for mark in marks:
            state = getattr(mark, "state", "") or ""
            if state.lower() != "selected":
                continue
            polygon = getattr(mark, "polygon", None)
            if not polygon or len(polygon) < 4:
                continue

            mcx, mcy = _poly_center(polygon)
            row_tolerance = _poly_height(polygon) * 1.5

            nearby: list[tuple[float, str]] = []
            for word in words:
                wpoly = getattr(word, "polygon", None)
                if not wpoly or len(wpoly) < 4:
                    continue
                wcx, wcy = _poly_center(wpoly)
                if abs(wcy - mcy) <= row_tolerance:
                    horizontal_dist = abs(wcx - mcx)
                    nearby.append((horizontal_dist, word.content))

            nearby.sort(key=lambda t: t[0])

            # Log the top candidates so we can see what's being picked and why.
            log.info(
                "ocr.spatial_debug   SELECTED mark center=(%.3f,%.3f) top_candidates=%s",
                mcx, mcy,
                [(round(d, 3), w) for d, w in nearby[:10]],
            )

            if nearby:
                label = _resolve_label(nearby)
                selected_labels.append(label)
                log.info("ocr.selected_mark label=%r", label)

    return selected_labels
