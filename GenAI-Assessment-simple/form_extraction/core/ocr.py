"""Run Azure Document Intelligence (prebuilt-layout) on an uploaded file.

The OCR stage is spatial-first: every structured field (dates, ID, phones,
checkboxes) is extracted from the raw polygon coordinates returned by Azure
Document Intelligence — not from the Markdown text stream, which suffers from
RTL reading-order quirks.

The LLM prompt therefore receives a compact, self-describing structured
header at the top ("=== FORM 283 SPATIAL EXTRACTION ==="), followed by the
Markdown OCR as a secondary free-text source for descriptive fields
(names, addresses, accident description, etc.).

Key spatial invariants relied on throughout
-------------------------------------------
* Every word has a polygon expressed as  [x0, y0, x1, y1, x2, y2, x3, y3]
  in inches. X grows LEFT→RIGHT; Y grows TOP→BOTTOM.
* Hebrew digit sequences inside boxed fields render LEFT→RIGHT visually even
  in RTL documents — so sorting words by X-centre always yields the correct
  digit order regardless of Azure's text reading direction.
* In Hebrew "☐ label" checkbox pairs, the LABEL is visually to the LEFT of
  its checkbox (RTL reading order: box first → label second → box to the
  right of label).  Our checkbox label resolver therefore prefers left-side
  labels when a selected mark sits between two candidate labels.
"""

from __future__ import annotations

import logging
import re
import time

from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import AnalyzeDocumentRequest, DocumentContentFormat
from azure.core.credentials import AzureKeyCredential

from form_extraction.core.config import Settings, get_settings

log = logging.getLogger("form_extraction.ocr")


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_ocr(data: bytes, settings: Settings | None = None) -> str:
    """Analyse the document and return OCR text annotated with spatial data.

    The returned string is assembled in three layers:

    1. ``=== FORM 283 SPATIAL EXTRACTION ===`` — authoritative key/value
       pairs produced by bounding-box analysis.  One line per field, each
       with the label's (x, y) coordinates and the extracted value.  The
       LLM should treat these as authoritative for every digit / checkbox
       field.

    2. ``=== SELECTED CHECKBOXES ===`` — one line per selected checkbox
       with the resolved label (picked via directional spatial matching).

    3. ``=== FORM BODY (markdown OCR) ===`` — the raw Azure DI Markdown
       stream, lightly processed (spaced-digit collapse + section banners).
       Used only for descriptive free-text fields the LLM still needs to
       read from context.
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
            locale="he",
            output_content_format=DocumentContentFormat.MARKDOWN,
        )
        result = poller.result()

    content = (result.content or "").strip()
    elapsed_ms = int((time.perf_counter() - t0) * 1000)
    if not content:
        log.warning("ocr.empty elapsed_ms=%d", elapsed_ms)
        raise RuntimeError("OCR returned empty content; the document may be blank or unreadable.")

    all_words = [w for page in (result.pages or []) for w in (page.words or [])]

    # --- Spatial extraction of every digit / checkbox field --------------
    spatial = extract_spatial_fields(result, all_words)

    # --- Markdown post-processing ---------------------------------------
    content, n_collapsed = _collapse_spaced_digits(content)
    if n_collapsed:
        log.info("ocr.collapsed_digit_sequences count=%d", n_collapsed)
    content = _segment_form_sections(content)

    output = _render_spatial_header(spatial) + "\n=== FORM BODY (markdown OCR) ===\n" + content

    log.info(
        "ocr.done chars=%d elapsed_ms=%d  receipt=%r filling=%r injury=%r birth=%r id=%r",
        len(output), elapsed_ms,
        spatial["receipt_date"], spatial["filling_date"], spatial["injury_date"],
        spatial["birth_date"], spatial["id_number"],
    )
    return output


# ---------------------------------------------------------------------------
# Spatial extraction — the core of this module
# ---------------------------------------------------------------------------

def extract_spatial_fields(result, all_words: list) -> dict:
    """Run every spatial extractor and return a dict suitable for rendering.

    Keys:
        receipt_date / filling_date / injury_date / birth_date  → DDMMYYYY
        id_number                                               → string of digits
        mobile_phone / landline_phone                           → string of digits
        selected_checkboxes                                     → list[dict] with label/pos
        health_fund                                             → one of "כללית/מכבי/מאוחדת/לאומית" or ""
        label_positions                                         → dict of label→(x,y)
    """
    # Build a cache of label positions we reference in the rendered header.
    label_pos: dict[str, tuple[float, float] | None] = {
        "receipt":  _find_word_pos(all_words, "בקופה"),
        "filling":  _find_word_pos(all_words, "מילוי"),
        "injury":   _find_word_pos(all_words, "הפגיעה"),
        "birth":    _find_word_pos(all_words, "לידה"),
        "id":       _find_id_label_pos(all_words),
        "landline": _find_phone_label_pos(all_words, "קווי"),
        "mobile":   _find_phone_label_pos(all_words, "נייד"),
    }

    # Dates — each has a UNIQUE keyword, so no exclusion dance required.
    receipt_date = _extract_date_below(all_words, "בקופה")
    filling_date = _extract_date_below(all_words, "מילוי")
    injury_date  = _extract_date_below(all_words, "הפגיעה")
    birth_date   = _extract_date_below(all_words, "לידה")

    id_number = _extract_id(all_words)
    mobile_phone   = _extract_phone_below(all_words, "נייד")
    landline_phone = _extract_phone_below(all_words, "קווי")

    selected = _extract_selected_labels(result)
    health_fund = _resolve_health_fund(selected)

    return {
        "receipt_date":  receipt_date,
        "filling_date":  filling_date,
        "injury_date":   injury_date,
        "birth_date":    birth_date,
        "id_number":     id_number,
        "mobile_phone":  mobile_phone,
        "landline_phone": landline_phone,
        "selected_checkboxes": selected,
        "health_fund":   health_fund,
        "label_positions": label_pos,
    }


# ---------------------------------------------------------------------------
# Rendering the spatial header that precedes the Markdown body
# ---------------------------------------------------------------------------

def _fmt_date(ddmmyyyy: str) -> str:
    if len(ddmmyyyy) == 8 and ddmmyyyy.isdigit():
        return f"{ddmmyyyy}  (day={ddmmyyyy[:2]}  month={ddmmyyyy[2:4]}  year={ddmmyyyy[4:]})"
    return "(not found)"


def _fmt_pos(pos: tuple[float, float] | None) -> str:
    return f"x={pos[0]:.2f} y={pos[1]:.2f}" if pos else "pos=?"


def _render_spatial_header(spatial: dict) -> str:
    L = spatial["label_positions"]
    lines: list[str] = []
    lines.append("=== FORM 283 SPATIAL EXTRACTION ===")
    lines.append("")
    lines.append("These values are derived from Azure Document Intelligence polygon")
    lines.append("coordinates (not from the RTL text stream).  They are authoritative")
    lines.append("for every dated / numbered / checkbox field on Form 283.  Use them")
    lines.append("verbatim and DO NOT re-derive from the markdown body below.")
    lines.append("")
    lines.append("-- Dates (8 digit-boxes, read LEFT→RIGHT by X coordinate) --")
    lines.append(f"formReceiptDateAtClinic: {_fmt_date(spatial['receipt_date'])}   [label 'תאריך קבלת הטופס בקופה' at {_fmt_pos(L['receipt'])}]")
    lines.append(f"formFillingDate:         {_fmt_date(spatial['filling_date'])}   [label 'תאריך מילוי הטופס' at {_fmt_pos(L['filling'])}]")
    lines.append(f"dateOfInjury:            {_fmt_date(spatial['injury_date'])}   [label 'תאריך הפגיעה' at {_fmt_pos(L['injury'])}]")
    lines.append(f"dateOfBirth:             {_fmt_date(spatial['birth_date'])}   [label 'תאריך לידה' at {_fmt_pos(L['birth'])}]")
    lines.append("")
    lines.append("-- Identifiers & phones (digit-boxes read LEFT→RIGHT) --")
    id_ = spatial["id_number"]
    lines.append(f"idNumber:                {id_ or '(not found)'}   [label 'ת.ז.' at {_fmt_pos(L['id'])}]")
    lines.append(f"mobilePhone:             {spatial['mobile_phone'] or '(not found)'}   [label 'טלפון נייד' at {_fmt_pos(L['mobile'])}]")
    lines.append(f"landlinePhone:           {spatial['landline_phone'] or '(not found)'}   [label 'טלפון קווי' at {_fmt_pos(L['landline'])}]")
    lines.append("")
    lines.append("-- Selected checkboxes (from Azure DI selection marks + directional label match) --")
    if spatial["selected_checkboxes"]:
        for mark in spatial["selected_checkboxes"]:
            lines.append(
                f"  [SELECTED] at ({mark['x']:.2f}, {mark['y']:.2f})  →  label: {mark['label']}"
            )
    else:
        lines.append("  (no selection marks detected)")
    lines.append("")
    lines.append(f"healthFundMember (resolved): {spatial['health_fund'] or '(none — no fund checkbox selected)'}")
    lines.append("")
    lines.append("Rules for the extractor:")
    lines.append("  • For every date / ID / phone / gender / accidentLocation /")
    lines.append("    healthFundMember field above, copy the value from this header.")
    lines.append("  • If the header says '(not found)' or '(none...)' the corresponding")
    lines.append("    field must be \"\" (empty string).")
    lines.append("  • Ignore ☐ / ☒ / :selected: markers in the markdown below — those")
    lines.append("    reflect the Azure DI text stream and are often misplaced in RTL.")
    lines.append("  • For free-text fields (names, address parts, job type, accident")
    lines.append("    description, injured body part, signature, clinic free text),")
    lines.append("    read from the markdown body below.")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Polygon helpers
# ---------------------------------------------------------------------------

def _poly_center(polygon: list[float]) -> tuple[float, float]:
    xs = polygon[0::2]
    ys = polygon[1::2]
    return sum(xs) / len(xs), sum(ys) / len(ys)


def _poly_height(polygon: list[float]) -> float:
    ys = polygon[1::2]
    return max(ys) - min(ys) if ys else 1.0


def _poly_bottom(polygon: list[float]) -> float:
    ys = polygon[1::2]
    return max(ys) if ys else 0.0


def _find_word_pos(all_words: list, keyword: str) -> tuple[float, float] | None:
    """Return the (cx, cy) of the first word containing *keyword*, else None."""
    for w in all_words:
        if keyword in (w.content or ""):
            poly = getattr(w, "polygon", None)
            if poly and len(poly) >= 4:
                return _poly_center(poly)
    return None


def _find_id_label_pos(all_words: list) -> tuple[float, float] | None:
    for w in all_words:
        c = (w.content or "").strip().replace(".", "").replace(" ", "")
        if c == "תז":
            poly = getattr(w, "polygon", None)
            if poly and len(poly) >= 4:
                return _poly_center(poly)
    return None


def _find_phone_label_pos(all_words: list, distinctive: str) -> tuple[float, float] | None:
    """Return position of the 'טלפון …' label with the given distinctive word."""
    # Prefer a word whose content literally equals 'טלפון' and which sits on
    # the same row as the distinctive keyword.  Falls back to the distinctive
    # keyword's own centre if pairing fails.
    telephone_words = [w for w in all_words if (w.content or "").strip() == "טלפון"]
    distinctive_words = [
        w for w in all_words
        if distinctive in (w.content or "") and w.content.strip() != "טלפון"
    ]
    best: tuple[float, float] | None = None
    best_dist: float = 1e9
    for tw in telephone_words:
        tpoly = getattr(tw, "polygon", None)
        if not tpoly or len(tpoly) < 4:
            continue
        tcx, tcy = _poly_center(tpoly)
        for dw in distinctive_words:
            dpoly = getattr(dw, "polygon", None)
            if not dpoly or len(dpoly) < 4:
                continue
            dcx, dcy = _poly_center(dpoly)
            if abs(dcy - tcy) < 0.3 and abs(dcx - tcx) < 2.0:
                d = abs(dcx - tcx) + abs(dcy - tcy)
                if d < best_dist:
                    best_dist = d
                    best = (tcx, tcy)
    if best is not None:
        return best
    # Fallback: distinctive word's own centre
    for dw in distinctive_words:
        dpoly = getattr(dw, "polygon", None)
        if dpoly and len(dpoly) >= 4:
            return _poly_center(dpoly)
    return None


# ---------------------------------------------------------------------------
# Digit-box region collector
# ---------------------------------------------------------------------------

def _collect_digits_in_region(
    all_words: list,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    reverse_multidigit_words: bool = False,
) -> str:
    """Return digits from words whose centre lies inside the region.

    Words are sorted by X (LEFT→RIGHT) so the concatenated string is always
    in visual reading order regardless of Azure's text-stream order.
    """
    hits: list[tuple[float, str]] = []
    for w in all_words:
        poly = getattr(w, "polygon", None)
        if not poly or len(poly) < 4:
            continue
        wcx, wcy = _poly_center(poly)
        if x_min <= wcx <= x_max and y_min <= wcy <= y_max:
            digits = re.sub(r"\D", "", w.content or "")
            if digits:
                if reverse_multidigit_words and len(digits) > 1:
                    digits = digits[::-1]
                hits.append((wcx, digits))
    hits.sort(key=lambda t: t[0])
    return "".join(d for _, d in hits)


# ---------------------------------------------------------------------------
# Date extraction (per-label)
# ---------------------------------------------------------------------------

def _extract_date_below(
    all_words: list,
    label_keyword: str,
    x_half_width: float = 1.3,
    y_search_inches: float = 1.0,
) -> str:
    """Extract a DDMMYYYY date from the digit-box row below *label_keyword*.

    The X window is intentionally narrow (±1.3 inches from the label word's
    centre by default) so neighbouring date columns at the top of the form
    are not picked up.  This is the key fix for the formReceiptDateAtClinic
    / formFillingDate ambiguity — prior ±3.0" window combined digits from
    both columns and parsed the FIRST valid 8-digit substring, which was
    the wrong column.
    """
    label_word = None
    for w in all_words:
        if label_keyword not in (w.content or ""):
            continue
        poly = getattr(w, "polygon", None)
        if not poly or len(poly) < 4:
            continue
        label_word = w
        break

    if label_word is None:
        log.debug("ocr.date.no_label keyword=%r", label_keyword)
        return ""

    poly = label_word.polygon
    lcx, lcy = _poly_center(poly)
    lbottom = _poly_bottom(poly)
    lh = _poly_height(poly)

    raw = _collect_digits_in_region(
        all_words,
        x_min=lcx - x_half_width,
        x_max=lcx + x_half_width,
        y_min=lbottom + lh * 0.1,
        y_max=lbottom + y_search_inches,
    )
    parsed = _parse_ddmmyyyy(raw)
    log.info("ocr.date keyword=%r label=(%.2f,%.2f) raw=%r parsed=%r",
             label_keyword, lcx, lcy, raw, parsed)
    return parsed


def _is_valid_date(day: int, month: int, year: int) -> bool:
    return 1 <= day <= 31 and 1 <= month <= 12 and 1900 <= year <= 2100


def _parse_ddmmyyyy(digits: str) -> str:
    """Try each 8-digit window of *digits* as DDMMYYYY.

    If the sequence has exactly 9 digits and no 8-digit window is a valid
    calendar date, try dropping one digit at each of the 9 positions —
    this recovers from the occasional Azure DI OCR hiccup that duplicates
    or hallucinates a single digit inside a digit-box row (e.g. ex2's
    injury date comes out as "120182005" instead of "12082005").
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
# Phone extraction
# ---------------------------------------------------------------------------

def _extract_phone_below(all_words: list, distinctive: str) -> str:
    """Extract the phone-number digit sequence below a 'טלפון …' label.

    The phone boxes sit directly beneath the two-word label ``טלפון נייד`` /
    ``טלפון קווי``.  We locate the distinctive word, then collect digits in
    a wide row beneath it (Israeli phones are 9 – 10 digits, roughly 2
    inches of boxes).
    """
    pos = _find_phone_label_pos(all_words, distinctive)
    if pos is None:
        log.debug("ocr.phone.no_label distinctive=%r", distinctive)
        return ""

    lcx, lcy = pos
    # Use a wider X window than dates because phone numbers are longer and
    # the phone label is NOT adjacent to another label in the same row.
    raw = _collect_digits_in_region(
        all_words,
        x_min=lcx - 2.0,
        x_max=lcx + 2.0,
        y_min=lcy + 0.05,
        y_max=lcy + 0.9,
    )
    # Keep only 9 – 10 digit sequences (Israeli phone lengths).
    if 8 <= len(raw) <= 12:
        log.info("ocr.phone distinctive=%r raw=%r", distinctive, raw)
        return raw
    log.info("ocr.phone distinctive=%r raw=%r (rejected, unusual length)", distinctive, raw)
    return raw  # still return so the LLM can see it


# ---------------------------------------------------------------------------
# ID extraction
# ---------------------------------------------------------------------------

def _extract_id(all_words: list) -> str:
    """Extract the ID by reading digit-boxes to the LEFT of the ת.ז. label.

    On Form 283 the ID digit-boxes are arranged in a horizontal row
    immediately to the left of the ת.ז. label.  Reading them by X
    coordinate (LEFT→RIGHT) gives the correct order directly with no bidi
    correction needed.
    """
    label_word = None
    for w in all_words:
        content = (w.content or "").strip().replace(".", "").replace(" ", "")
        if content == "תז":
            poly = getattr(w, "polygon", None)
            if poly and len(poly) >= 4:
                label_word = w
                break

    if label_word is None:
        log.warning("ocr.id.no_label")
        return ""

    poly = label_word.polygon
    lcx, lcy = _poly_center(poly)
    lh = _poly_height(poly)

    digits = _collect_digits_in_region(
        all_words,
        x_min=lcx - 4.0,
        x_max=lcx - 0.05,
        y_min=lcy - lh * 3,
        y_max=lcy + lh * 3,
    )

    log.info("ocr.id label=(%.2f,%.2f) digits=%r", lcx, lcy, digits)
    return digits


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
       fallback for cases where Azure's polygon for the label is slightly
       mis-placed (mis-shifted right of its visual location).

    3. Multi-word phrases: ``הנפגע [אינו] חבר בקופת חולים`` /
       ``בדרך לעבודה`` / ``בדרך מהעבודה`` / ``ת. דרכים בעבודה`` /
       ``מחוץ למפעל``.  These are recognised by the set of nearby words.

    4. Last-resort: the three closest words concatenated.
    """
    word_set = {w for _, _, w in candidates}

    # Stage 1 — single-label, LEFT side, ≤0.8"
    left_singles = [
        (abs(sdx), word)
        for sdx, _, word in candidates
        if sdx < 0 and abs(sdx) <= 0.8 and word in _SINGLE_LABELS
    ]
    if left_singles:
        left_singles.sort(key=lambda t: t[0])
        return left_singles[0][1]

    # Stage 2 — single-label, RIGHT side, ≤0.5"
    right_singles = [
        (abs(sdx), word)
        for sdx, _, word in candidates
        if sdx > 0 and abs(sdx) <= 0.5 and word in _SINGLE_LABELS
    ]
    if right_singles:
        right_singles.sort(key=lambda t: t[0])
        return right_singles[0][1]

    # Stage 3 — multi-word phrases
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

    # Stage 4 — last-resort concatenation
    candidates_sorted = sorted(candidates, key=lambda t: t[1])
    return " ".join(w for _, _, w in candidates_sorted[:3])


def _extract_selected_labels(result) -> list[dict]:
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
# Digit-box collapse (Markdown body only)
# ---------------------------------------------------------------------------

_SPACED_DIGIT_RE = re.compile(
    r"\[?"
    r"\d[\s\[\]|]*\d[\s\[\]|]*\d[\s\[\]|]*\d"
    r"[\s\[\]|]*"
    r"\d[\s\[\]|]*\d[\s\[\]|]*\d[\s\[\]|]*\d"
)


def _collapse_spaced_digits(text: str) -> tuple[str, int]:
    count = 0

    def _replace(m: re.Match) -> str:
        nonlocal count
        digits = re.sub(r"\D", "", m.group(0))
        if len(digits) == 8:
            count += 1
            return digits
        return m.group(0)

    return _SPACED_DIGIT_RE.sub(_replace, text), count


# ---------------------------------------------------------------------------
# Section segmentation (Markdown body only — cosmetic aid for the LLM)
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
