"""Run Azure Document Intelligence (prebuilt-layout) on an uploaded file.

The OCR stage produces a self-describing document with two parts:

  1. ``=== FORM 283 SPATIAL EXTRACTION ===`` — pre-computed key/value pairs
     for every date, ID, phone, and checkbox field on Form 283.

     Structured fields (dates, ID, phones) are extracted by collecting all
     Azure DI *words* whose centre falls inside a pre-defined bounding-box
     region for that field (see ``field_regions.py``).  Because Form 283 is
     a fixed-layout government form, the field positions are stable across
     every printed copy; we only need to calibrate the regions once against
     the blank form (see ``calibrate.py``).

     Checkboxes are extracted from the polygon-based selection marks,
     because the ☐/☒ symbols in the Markdown stream are frequently
     misplaced under RTL reordering.

  2. ``=== FORM BODY (markdown OCR) ===`` — the raw Markdown stream,
     lightly processed (spaced-digit collapse + section banners).  The
     LLM uses this only for descriptive free-text fields (names, address
     parts, job type, accident description, injured body part, signature,
     clinic free text).

Design rationale
----------------
The previous approach used label-anchored regex on the Markdown stream to
locate structured fields.  That worked well but required RTL-fallback
heuristics for cases where Azure DI's reading-order reflow moved a Hebrew
label away from its value.

The coordinate-based approach eliminates that fragility entirely: we look
directly at the geometry of each word returned by Azure DI, ask "is the
centre of this word inside the known input region for field X?", and
collect all matching words.  No label matching, no fallbacks, no
text-order assumptions.
"""

from __future__ import annotations

import logging
import re
import time

from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import AnalyzeDocumentRequest, DocumentContentFormat
from azure.core.credentials import AzureKeyCredential

from form_extraction.core.config import Settings, get_settings
from form_extraction.core.field_regions import PAGE_1

log = logging.getLogger("form_extraction.ocr")


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_ocr(data: bytes, settings: Settings | None = None) -> str:
    """Analyse the document and return OCR text annotated with spatial data."""
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
            locale="he",
            output_content_format=DocumentContentFormat.MARKDOWN,
        )
        result = poller.result()

    content = (result.content or "").strip()
    elapsed_ms = int((time.perf_counter() - t0) * 1000)
    if not content:
        log.warning("ocr.empty elapsed_ms=%d", elapsed_ms)
        raise RuntimeError("OCR returned empty content; the document may be blank or unreadable.")

    # --- Markdown preprocessing — section banners to aid LLM navigation ------
    content = _segment_form_sections(content)

    # --- Extract structured fields from Azure DI word coordinates ------------
    coord_fields = _extract_coordinate_fields(result)

    # --- Extract checkboxes from polygon data (authoritative for ☐/☒) ------
    selected = _extract_selected_checkboxes(result)
    health_fund = _resolve_health_fund(selected)

    # --- Render the spatial header ------------------------------------------
    header = _render_spatial_header(coord_fields, selected, health_fund)

    output = header + "\n=== FORM BODY (markdown OCR) ===\n" + content

    log.info(
        "ocr.done chars=%d elapsed_ms=%d  receipt=%r filling=%r injury=%r birth=%r id=%r mobile=%r landline=%r fund=%r",
        len(output), elapsed_ms,
        coord_fields["receipt_date"], coord_fields["filling_date"],
        coord_fields["injury_date"],  coord_fields["birth_date"],
        coord_fields["id_number"],    coord_fields["mobile_phone"],
        coord_fields["landline_phone"], health_fund,
    )
    return output


# ---------------------------------------------------------------------------
# Coordinate-based field extraction
# ---------------------------------------------------------------------------
#
# Every structured field on Form 283 has a fixed position on the page.
# We define each field's input area as a bounding box (x0, y0, x1, y1)
# in inches (see field_regions.py).  To extract a field we:
#
#   1. Iterate over every word Azure DI found on that page.
#   2. Compute the word's centre point from its polygon.
#   3. Keep the word if its centre falls inside the field's bounding box.
#   4. Sort kept words into reading order and join them.
#
# This is structurally identical to the existing checkbox extraction, which
# already uses polygon geometry — just applied to text words instead of
# selection marks.

_ROW_TOLERANCE = 0.12   # inches — words within this vertical distance are on the same row


def _poly_center(polygon: list[float]) -> tuple[float, float]:
    xs = polygon[0::2]
    ys = polygon[1::2]
    return sum(xs) / len(xs), sum(ys) / len(ys)


def _extract_field_text(
    page,
    region: tuple[float, float, float, float],
    rtl: bool = True,
) -> str:
    """Return the text of all words whose centre falls inside *region*.

    Parameters
    ----------
    page:
        An Azure DI page object (``result.pages[i]``).
    region:
        ``(x0, y0, x1, y1)`` in inches.
    rtl:
        If ``True``, words within each row are sorted right-to-left
        (correct for Hebrew prose).  For purely numeric fields the sort
        direction is irrelevant — pass ``rtl=False`` to keep LTR order.
    """
    x0, y0, x1, y1 = region
    collected: list[tuple[float, float, str]] = []

    for word in (page.words or []):
        poly = getattr(word, "polygon", None)
        if not poly or len(poly) < 4:
            continue
        cx, cy = _poly_center(poly)
        if x0 <= cx <= x1 and y0 <= cy <= y1:
            collected.append((cy, cx, (word.content or "").strip()))

    if not collected:
        return ""

    # Group words into horizontal rows.
    collected.sort(key=lambda w: w[0])  # top-to-bottom first
    rows: list[list[tuple[float, float, str]]] = []
    current_row: list[tuple[float, float, str]] = [collected[0]]
    for word in collected[1:]:
        if abs(word[0] - current_row[-1][0]) <= _ROW_TOLERANCE:
            current_row.append(word)
        else:
            rows.append(current_row)
            current_row = [word]
    rows.append(current_row)

    parts: list[str] = []
    for row in rows:
        # Sort within each row: right-to-left for Hebrew, left-to-right for numbers.
        row.sort(key=lambda w: -w[1] if rtl else w[1])
        parts.append(" ".join(w[2] for w in row))

    return " ".join(parts).strip()


def _extract_digit_field(page, region: tuple[float, float, float, float]) -> str:
    """Extract all words in *region* and return only the digit characters."""
    raw = _extract_field_text(page, region, rtl=False)
    return re.sub(r"\D", "", raw)


def _extract_date_field(page, region: tuple[float, float, float, float]) -> str:
    """Extract a date field and return DDMMYYYY (or '' if not found/parseable)."""
    digits = _extract_digit_field(page, region)
    if not digits:
        return ""
    parsed = _parse_ddmmyyyy(digits)
    if parsed:
        return parsed
    # If we have at least 8 digits but _parse_ddmmyyyy couldn't validate the
    # calendar date (e.g. a partially filled form), return the raw 8 digits so
    # the LLM can still see something rather than a blank.
    return digits[:8] if len(digits) >= 8 else ""


def _extract_coordinate_fields(result) -> dict:
    """Extract every structured field on page 1 using bounding-box regions.

    Returns a dict with the same keys as the old ``_extract_markdown_fields``
    so the rest of the pipeline (render_spatial_header, logging) is unchanged.
    """
    if not result.pages:
        log.warning("ocr.coord.no_pages")
        return {k: "" for k in (
            "receipt_date", "filling_date", "injury_date", "birth_date",
            "id_number", "mobile_phone", "landline_phone",
        )}

    page = result.pages[0]  # all structured fields are on page 1

    fields = {
        "receipt_date":   _extract_date_field(page,  PAGE_1["receipt_date"]),
        "filling_date":   _extract_date_field(page,  PAGE_1["filling_date"]),
        "injury_date":    _extract_date_field(page,  PAGE_1["injury_date"]),
        "birth_date":     _extract_date_field(page,  PAGE_1["birth_date"]),
        "id_number":      _extract_digit_field(page, PAGE_1["id_number"]),
        "mobile_phone":   _extract_digit_field(page, PAGE_1["mobile_phone"]),
        "landline_phone": _extract_digit_field(page, PAGE_1["landline_phone"]),
    }

    log.info(
        "ocr.coord.fields receipt=%r filling=%r injury=%r birth=%r id=%r mobile=%r landline=%r",
        fields["receipt_date"], fields["filling_date"], fields["injury_date"],
        fields["birth_date"], fields["id_number"], fields["mobile_phone"],
        fields["landline_phone"],
    )
    return fields


# ---------------------------------------------------------------------------
# Date parsing (with 9-digit recovery)
# ---------------------------------------------------------------------------

def _is_valid_date(day: int, month: int, year: int) -> bool:
    return 1 <= day <= 31 and 1 <= month <= 12 and 1900 <= year <= 2100


def _parse_ddmmyyyy(digits: str) -> str:
    """Try each 8-digit window of *digits* as DDMMYYYY.

    If the sequence has exactly 9 digits and no 8-digit window is a
    valid calendar date, try dropping one digit at each of the 9
    positions — this recovers from the occasional Azure DI hiccup that
    inserts an extra digit inside a digit-box row.
    """
    if not digits:
        return ""

    for i in range(max(0, len(digits) - 7)):
        w = digits[i: i + 8]
        if len(w) < 8:
            break
        try:
            if _is_valid_date(int(w[:2]), int(w[2:4]), int(w[4:])):
                return w
        except ValueError:
            continue

    if len(digits) == 9:
        for skip in range(9):
            w = digits[:skip] + digits[skip + 1:]
            try:
                if _is_valid_date(int(w[:2]), int(w[2:4]), int(w[4:])):
                    return w
            except ValueError:
                continue

    return ""


# ---------------------------------------------------------------------------
# Rendering the spatial header that precedes the Markdown body
# ---------------------------------------------------------------------------

def _fmt_date(ddmmyyyy: str) -> str:
    if len(ddmmyyyy) == 8 and ddmmyyyy.isdigit():
        return f"{ddmmyyyy}  (day={ddmmyyyy[:2]}  month={ddmmyyyy[2:4]}  year={ddmmyyyy[4:]})"
    return "(not found)"


def _render_spatial_header(
    coord_fields: dict,
    selected: list[dict],
    health_fund: str,
) -> str:
    lines: list[str] = []
    lines.append("=== FORM 283 SPATIAL EXTRACTION ===")
    lines.append("")
    lines.append("These values are pre-computed from Azure DI word coordinates.")
    lines.append("Each field was extracted by collecting all words whose centre")
    lines.append("falls inside a calibrated bounding-box region for that field.")
    lines.append("Checkboxes are resolved from polygon-based selection marks.")
    lines.append("They are authoritative — use them verbatim and do NOT")
    lines.append("re-derive from the markdown body below.")
    lines.append("")
    lines.append("-- Dates (DDMMYYYY, extracted from fixed coordinate regions) --")
    lines.append(f"formReceiptDateAtClinic: {_fmt_date(coord_fields['receipt_date'])}")
    lines.append(f"formFillingDate:         {_fmt_date(coord_fields['filling_date'])}")
    lines.append(f"dateOfInjury:            {_fmt_date(coord_fields['injury_date'])}")
    lines.append(f"dateOfBirth:             {_fmt_date(coord_fields['birth_date'])}")
    lines.append("")
    lines.append("-- Identifiers & phones --")
    lines.append(f"idNumber:                {coord_fields['id_number'] or '(not found)'}")
    lines.append(f"mobilePhone:             {coord_fields['mobile_phone'] or '(not found)'}")
    lines.append(f"landlinePhone:           {coord_fields['landline_phone'] or '(not found)'}")
    lines.append("")
    lines.append("-- Selected checkboxes (from Azure DI selection marks + directional label match) --")
    if selected:
        for mark in selected:
            lines.append(
                f"  [SELECTED] at ({mark['x']:.2f}, {mark['y']:.2f})  →  label: {mark['label']}"
            )
    else:
        lines.append("  (no selection marks detected)")
    lines.append("")
    lines.append(f"healthFundMember (resolved): {health_fund or '(none — no fund checkbox selected)'}")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Polygon helpers (shared by coordinate extractor and checkbox extractor)
# ---------------------------------------------------------------------------

def _poly_height(polygon: list[float]) -> float:
    ys = polygon[1::2]
    return max(ys) - min(ys) if ys else 1.0


# ---------------------------------------------------------------------------
# Checkbox extraction & directional label resolution
# ---------------------------------------------------------------------------

_SINGLE_LABELS: frozenset[str] = frozenset({
    "זכר", "נקבה",
    "כללית", "מכבי", "מאוחדת", "לאומית",
    "במפעל", "אחר",
})

_FUND_NAMES: frozenset[str] = frozenset({"כללית", "מכבי", "מאוחדת", "לאומית"})


def _resolve_label_directional(candidates: list[tuple[float, float, str]]) -> str:
    """Choose the label for a selected checkbox.

    Arguments
    ---------
    candidates : list of (signed_dx, abs_dx, word)
        signed_dx = word_centre_x − mark_centre_x
        (negative = word is to the LEFT of the mark; positive = RIGHT).

    Selection preference order
    --------------------------
    1. Single whitelist label (זכר / נקבה / fund name / במפעל / אחר)
       that is **to the LEFT** of the mark and within 0.8" — Hebrew RTL
       convention pairs the label with the checkbox on its RIGHT.
       Among qualifying words, the one closest to the mark wins.

    2. Single whitelist label to the RIGHT of the mark, within 0.5" —
       fallback for cases where Azure's polygon for the label is
       slightly misplaced.

    3. Multi-word phrases: ``הנפגע [אינו] חבר בקופת חולים`` /
       ``בדרך לעבודה`` / ``בדרך מהעבודה`` / ``ת. דרכים בעבודה`` /
       ``מחוץ למפעל``.

    4. Last-resort: the three closest words concatenated.
    """
    word_set = {w for _, _, w in candidates}

    left_singles = [
        (abs(sdx), word)
        for sdx, _, word in candidates
        if sdx < 0 and abs(sdx) <= 0.8 and word in _SINGLE_LABELS
    ]
    if left_singles:
        left_singles.sort(key=lambda t: t[0])
        return left_singles[0][1]

    right_singles = [
        (abs(sdx), word)
        for sdx, _, word in candidates
        if sdx > 0 and abs(sdx) <= 0.5 and word in _SINGLE_LABELS
    ]
    if right_singles:
        right_singles.sort(key=lambda t: t[0])
        return right_singles[0][1]

    if "הנפגע" in word_set:
        if "אינו" in word_set:
            return "הנפגע אינו חבר בקופת חולים"
        return "הנפגע חבר בקופת חולים"

    if "בדרך" in word_set:
        if "לעבודה" in word_set:
            return "בדרך לעבודה"
        if "מהעבודה" in word_set:
            return "בדרך מהעבודה"

    if "ת." in word_set and "דרכים" in word_set:
        return "ת. דרכים בעבודה"

    if "מחוץ" in word_set:
        return "מחוץ למפעל"

    candidates_sorted = sorted(candidates, key=lambda t: t[1])
    return " ".join(w for _, _, w in candidates_sorted[:3])


def _extract_selected_checkboxes(result) -> list[dict]:
    """Return one dict per selected selection mark."""
    out: list[dict] = []

    for page_idx, page in enumerate(result.pages or []):
        marks = page.selection_marks or []
        words = page.words or []

        log.info(
            "ocr.checkbox page=%d marks=%d words=%d",
            page_idx + 1, len(marks), len(words),
        )

        for mark in marks:
            state = getattr(mark, "state", "") or ""
            if state.lower() != "selected":
                continue
            polygon = getattr(mark, "polygon", None)
            if not polygon or len(polygon) < 4:
                continue

            mcx, mcy = _poly_center(polygon)
            row_tol = max(_poly_height(polygon) * 2.5, 0.15)

            candidates: list[tuple[float, float, str]] = []
            for word in words:
                wpoly = getattr(word, "polygon", None)
                if not wpoly or len(wpoly) < 4:
                    continue
                wcx, wcy = _poly_center(wpoly)
                if abs(wcy - mcy) <= row_tol:
                    signed = wcx - mcx
                    candidates.append((signed, abs(signed), (word.content or "").strip()))

            candidates.sort(key=lambda t: t[1])
            label = _resolve_label_directional(candidates[:20]) if candidates else "(no label)"
            log.info(
                "ocr.checkbox.sel mark=(%.2f,%.2f) row_tol=%.2f top=%s  →  label=%r",
                mcx, mcy, row_tol,
                [(round(sdx, 2), w) for sdx, _, w in candidates[:6]],
                label,
            )
            out.append({"x": mcx, "y": mcy, "label": label})

    return out


def _resolve_health_fund(selected: list[dict]) -> str:
    for mark in selected:
        if mark["label"] in _FUND_NAMES:
            return mark["label"]
    return ""


# ---------------------------------------------------------------------------
# Digit-box collapse (Markdown body preprocessing for LLM)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Section segmentation (Markdown body — cosmetic aid for the LLM)
# ---------------------------------------------------------------------------

_SECTION_MARKERS: list[tuple[re.Pattern, str]] = [
    (
        re.compile(r"^#\s*1\s*$", re.MULTILINE),
        "=== SECTION 1: תאריך הפגיעה (dateOfInjury) ===",
    ),
    (
        re.compile(r"^2\s*$", re.MULTILINE),
        "=== SECTION 2: פרטי התובע (lastName, firstName, idNumber, gender, dateOfBirth, address, phones) ===",
    ),
    (
        re.compile(r"^3\s*$", re.MULTILINE),
        "=== SECTION 3: פרטי התאונה (timeOfInjury, jobType, accidentLocation, accidentAddress, accidentDescription, injuredBodyPart) ===",
    ),
    (
        re.compile(r"^#\s*4\s*$", re.MULTILINE),
        "=== SECTION 4: הצהרה (signature) ===",
    ),
    (
        re.compile(r"^(?:#{1,3}\s*)?5(?:\s.*)?$", re.MULTILINE),
        "=== SECTION 5: למילוי ע\"י המוסד הרפואי (natureOfAccident, medicalDiagnoses) ===",
    ),
]

_FORM_HEADER_BANNER = "=== FORM HEADER (formReceiptDateAtClinic, formFillingDate) ==="


def _segment_form_sections(content: str) -> str:
    for pattern, banner in _SECTION_MARKERS:
        content = pattern.sub(f"\n{banner}\n", content)

    sec1_banner = "=== SECTION 1:"
    idx = content.find(sec1_banner)
    if idx > 0:
        header_text = content[:idx].rstrip()
        rest = content[idx:]
        content = f"{_FORM_HEADER_BANNER}\n{header_text}\n\n{rest}"

    return content
