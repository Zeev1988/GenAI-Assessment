"""Run Azure Document Intelligence (prebuilt-layout) on an uploaded file.

The OCR stage produces a self-describing document with two parts:

  1. ``=== FORM 283 SPATIAL EXTRACTION ===`` — pre-computed key/value pairs
     for every date, ID, phone, and checkbox field on Form 283.  Dates,
     IDs and phones are extracted from the Markdown text stream (which
     Azure DI produces in logical reading order and — after spaced-digit
     collapse — is clean enough to anchor a regex on each label).
     Checkboxes are extracted from the polygon-based selection marks,
     because the ☐/☒ symbols in the Markdown stream are frequently
     misplaced under RTL reordering.

  2. ``=== FORM BODY (markdown OCR) ===`` — the raw Markdown stream,
     lightly processed (spaced-digit collapse + section banners).  The
     LLM uses this only for descriptive free-text fields (names,
     address parts, job type, accident description, injured body part,
     signature, clinic free text).

This layered approach keeps each part of the pipeline doing the thing
it does well: Azure DI does OCR + reading order + polygon geometry;
Python does label-anchored regex on a pre-cleaned text stream; the LLM
does natural-language understanding of the free-text sections.
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

    # --- Markdown preprocessing (collapse spaced digits + segment sections) --
    content, n_collapsed = _collapse_spaced_digits(content)
    if n_collapsed:
        log.info("ocr.collapsed_digit_sequences count=%d", n_collapsed)
    content = _segment_form_sections(content)

    # --- Extract structured fields from the PREPROCESSED markdown -----------
    md_fields = _extract_markdown_fields(content)

    # --- Extract checkboxes from polygon data (authoritative for ☐/☒) ------
    selected = _extract_selected_checkboxes(result)
    health_fund = _resolve_health_fund(selected)

    # --- Render the spatial header ------------------------------------------
    header = _render_spatial_header(md_fields, selected, health_fund)

    output = header + "\n=== FORM BODY (markdown OCR) ===\n" + content

    log.info(
        "ocr.done chars=%d elapsed_ms=%d  receipt=%r filling=%r injury=%r birth=%r id=%r mobile=%r landline=%r fund=%r",
        len(output), elapsed_ms,
        md_fields["receipt_date"], md_fields["filling_date"],
        md_fields["injury_date"], md_fields["birth_date"],
        md_fields["id_number"], md_fields["mobile_phone"],
        md_fields["landline_phone"], health_fund,
    )
    return output


# ---------------------------------------------------------------------------
# Markdown-based field extraction
# ---------------------------------------------------------------------------
#
# Form 283 is divided into well-defined sections (we insert banners during
# preprocessing).  Each dated / numbered field has:
#
#   • a distinctive Hebrew label that, when present, is the strongest
#     anchor we can hope for (e.g. "תאריך לידה" → birth date);
#   • a canonical section where the value normally lives;
#   • a positional fallback (N-th valid DDMMYYYY in that section) for
#     cases where RTL reading-order has moved the label out of the
#     section even though the value itself is still there.  ex3 is the
#     canonical example: its "תאריך לידה" label lands in SECTION 5 but
#     the birth-date digits are in SECTION 2.
#
# Both strategies are applied per field (label-first, positional-fallback).
# Adding or tweaking a field is a one-line change in ``_DATE_FIELDS``.

# Date fields: (output_key, label regex, canonical section key, fallback nth).
# The label regex uses ``\s*`` liberally — Azure DI sometimes collapses /
# inserts whitespace around the Hebrew words.  Anchoring on a single
# keyword (e.g. "הפגיעה") risks cross-matching another phrase, so we
# include two keywords per label where possible.
_DATE_FIELDS: list[tuple[str, str, str, int]] = [
    ("receipt_date", r"תאריך\s*קבלת\s*הטופס",  "header",   1),
    ("filling_date", r"תאריך\s*מילוי\s*הטופס", "header",   2),
    ("injury_date",  r"תאריך\s*הפגיעה",          "section1", 1),
    ("birth_date",   r"תאריך\s*לידה",            "section2", 1),
]

# How far after a date label to scan for the DDMMYYYY value.  Generous
# because forms often interleave checkboxes / "מין" label / table rows
# between the label and the date (see ex1/ex2 section 2).
_DATE_LOOKAHEAD = 400


def _extract_markdown_fields(content: str) -> dict:
    """Extract every dated / numbered field from the preprocessed markdown."""
    sections = _split_sections(content)

    # Per-field label-anchored-first-then-section-nth date extraction.
    dates: dict[str, str] = {}
    for key, label_pat, section_key, fallback_n in _DATE_FIELDS:
        dates[key] = _extract_date(
            sections.get(section_key, ""), label_pat, fallback_n
        )

    id_number = _extract_id(content)
    mobile, landline = _extract_phones(content)

    return {
        "receipt_date":   dates["receipt_date"],
        "filling_date":   dates["filling_date"],
        "injury_date":    dates["injury_date"],
        "birth_date":     dates["birth_date"],
        "id_number":      id_number,
        "mobile_phone":   mobile,
        "landline_phone": landline,
    }


# --- Section splitter -------------------------------------------------------

_SECTION_BANNERS: list[tuple[str, str]] = [
    ("header",   "=== FORM HEADER"),
    ("section1", "=== SECTION 1:"),
    ("section2", "=== SECTION 2:"),
    ("section3", "=== SECTION 3:"),
    ("section4", "=== SECTION 4:"),
    ("section5", "=== SECTION 5:"),
]


def _split_sections(content: str) -> dict[str, str]:
    """Split the preprocessed markdown into a dict of {section_name: text}.

    Banner positions are discovered in-order; each section extends up to
    the next banner (or end of content).  If a banner is missing the
    corresponding key is absent from the returned dict.
    """
    positions: list[tuple[int, str]] = []
    for name, banner in _SECTION_BANNERS:
        i = content.find(banner)
        if i >= 0:
            positions.append((i, name))
    positions.sort()

    out: dict[str, str] = {}
    for i, (start, name) in enumerate(positions):
        end = positions[i + 1][0] if i + 1 < len(positions) else len(content)
        out[name] = content[start:end]
    return out


# --- Date extraction (label-anchored first, section-n-th fallback) --------

def _extract_date(section_text: str, label_pattern: str, fallback_nth: int) -> str:
    """Return a DDMMYYYY string for one date field in ``section_text``.

    Strategy:

    1. If ``label_pattern`` matches somewhere in the section, look in a
       window of ``_DATE_LOOKAHEAD`` characters immediately after the
       match for the first run of 8-9 digits that parses as a valid
       calendar date.  A present-but-unfilled label correctly yields ""
       instead of silently falling back (which would pick up a
       neighbouring field's value).

    2. If the label is NOT present in the section at all (the RTL-reflow
       case — see ex3's birth-date label in SECTION 5), fall back to the
       ``fallback_nth``-th valid DDMMYYYY run in the whole section.

    This is strictly more robust than n-th-only because it does not
    depend on the order in which Azure DI happens to lay out multiple
    dates within the section.
    """
    if not section_text:
        return ""

    m = re.search(label_pattern, section_text)
    if m:
        start = m.end()
        window = section_text[start: start + _DATE_LOOKAHEAD]
        for mm in re.finditer(r"\d{8,9}", window):
            parsed = _parse_ddmmyyyy(mm.group(0))
            if parsed:
                log.info(
                    "ocr.md.date label=%r -> %r (via label anchor)",
                    label_pattern, parsed,
                )
                return parsed
        # Label is there but no date within window → field is genuinely
        # blank on this form.  Don't fall back to positional nth — doing
        # so would silently pick a neighbouring field's value.
        log.info("ocr.md.date label=%r found but no date within %d chars",
                 label_pattern, _DATE_LOOKAHEAD)
        return ""

    # Label absent from this section — RTL reflow may have moved it
    # elsewhere.  Fall back to n-th positional within the section.
    return _find_nth_valid_date(section_text, fallback_nth)


def _find_nth_valid_date(section: str, n: int) -> str:
    """Return the *n*-th (1-based) 8-digit DDMMYYYY in ``section``.

    Walks digit runs of 8-9 chars in order; each one is fed through
    :func:`_parse_ddmmyyyy` which (a) validates an 8-digit sequence as a
    real calendar date and (b) performs 9-digit drop-one-digit recovery
    for cases like ex2's ``120182005 → 12082005``.  Runs that don't yield
    a valid date are skipped.
    """
    if not section:
        return ""
    count = 0
    for m in re.finditer(r"\d{8,9}", section):
        parsed = _parse_ddmmyyyy(m.group(0))
        if not parsed:
            continue
        count += 1
        if count == n:
            log.info(
                "ocr.md.date section_len=%d nth=%d parsed=%r (via fallback)",
                len(section), n, parsed,
            )
            return parsed
    log.info("ocr.md.date section_len=%d nth=%d not_found", len(section), n)
    return ""


# --- Phone regions (used by both ID and phone extraction) ------------------
#
# A "phone region" is a character interval that begins at a "טלפון" label
# and extends to whichever comes first:
#     (a) the next "טלפון" label (the other phone slot on the form), or
#     (b) the next section banner ("=== …"), or
#     (c) a generous character cap.
#
# Digits found INSIDE a phone region are treated as phone candidates;
# digits OUTSIDE every region are candidates for the ID slot.  This
# replaces an older hand-tuned "60 chars upstream of the digits contains
# 'טלפון'" heuristic that silently fails on long gaps.

_PHONE_LABEL = "טלפון"
_PHONE_REGION_CAP = 400  # upper bound in chars when no banner/label is nearer
_BANNER_RE = re.compile(r"^===", re.MULTILINE)


def _phone_regions(content: str) -> list[tuple[int, int]]:
    """Return [(start, end), …] covering every 'טלפון' slot in ``content``."""
    phone_positions = [m.start() for m in re.finditer(_PHONE_LABEL, content)]
    banner_positions = [m.start() for m in _BANNER_RE.finditer(content)]

    regions: list[tuple[int, int]] = []
    for i, s in enumerate(phone_positions):
        # Bound by the next 'טלפון' label (other phone slot).
        next_phone = phone_positions[i + 1] if i + 1 < len(phone_positions) else None
        # Bound by the next section banner after the label.
        next_banner = next((b for b in banner_positions if b > s), None)
        candidates = [s + _PHONE_REGION_CAP]
        if next_phone is not None:
            candidates.append(next_phone)
        if next_banner is not None:
            candidates.append(next_banner)
        regions.append((s, min(candidates)))
    return regions


def _in_any_region(pos: int, regions: list[tuple[int, int]]) -> bool:
    return any(s <= pos < e for s, e in regions)


# --- ID extraction ---------------------------------------------------------

_ID_LABEL_RE = re.compile(r"ת\s*\.?\s*ז\s*\.?")


def _extract_id(content: str) -> str:
    """Extract the 9-10 digit Israeli ID from the form body.

    The ID's canonical home is section 2, but under RTL re-flow Azure DI
    sometimes drops it into a neighbouring section (ex2 → section 5).
    So we collect every 9-10-digit run in the body, reject runs that are
    clearly phones (inside a phone region) or dates, and pick the
    candidate closest to the ``ת.ז.`` label.
    """
    # Find the ID-label anchor position (Form 283 only has one).
    label_match = _ID_LABEL_RE.search(content)
    label_pos = label_match.end() if label_match else -1

    phone_regions = _phone_regions(content)

    candidates: list[tuple[int, int, str]] = []
    for m in re.finditer(r"\d{9,10}", content):
        start, end = m.start(), m.end()
        digits = m.group(0)

        # Reject 9-digit DDMMYYYY glitches (birth/injury/etc.)
        if len(digits) == 9 and _parse_ddmmyyyy(digits):
            continue

        # Reject digits physically sitting in a phone slot.
        if _in_any_region(start, phone_regions):
            continue

        candidates.append((start, end, digits))

    if not candidates:
        log.warning("ocr.md.id.not_found")
        return ""

    # Prefer the candidate closest to the label.
    if label_pos >= 0:
        candidates.sort(key=lambda c: abs(c[0] - label_pos))

    start, _, digits = candidates[0]
    # RTL-reverse if a Hebrew letter sits on the SAME LINE as the digits
    # (e.g. ex3's "עי 7651254330").  A Hebrew word on a *previous* line
    # (like the "ס״ב" checksum-label row) does NOT imply reversal.
    line_start = content.rfind("\n", 0, start) + 1
    prefix_same_line = content[line_start: start]
    reverse = bool(re.search(r"[\u0590-\u05FF]", prefix_same_line))
    value = digits[::-1] if reverse else digits
    log.info(
        "ocr.md.id raw=%r same_line_prefix=%r reverse=%s → %r",
        digits, prefix_same_line.strip(), reverse, value,
    )
    return value


# --- Phone extraction ------------------------------------------------------

def _extract_phones(content: str) -> tuple[str, str]:
    """Return (mobile, landline) — digits found after each label.

    Labels anchor the search: "טלפון נייד" → mobile, "טלפון קווי" →
    landline.  For each label we scan forward to the next phone region
    boundary (next 'טלפון' label or section banner, whichever is
    closer) and return the first 9-12-digit run found.  If the landline
    field is physically empty on the form (ex1) its window ends right
    before the mobile label, so no false-positive capture occurs.
    """
    mobile = _find_phone_near(content, "טלפון נייד")
    landline = _find_phone_near(content, "טלפון קווי")
    log.info("ocr.md.phones mobile=%r landline=%r", mobile, landline)
    return mobile, landline


def _find_phone_near(content: str, label: str) -> str:
    """Scan forward from *label* for a 9-12-digit phone-like run.

    The window ends at whichever comes first: the next 'טלפון' label,
    the next section banner, or ``_PHONE_REGION_CAP`` characters past
    the label.  This uses the same region boundary logic as
    :func:`_phone_regions`, so phone extraction and ID-rejection stay
    consistent with each other.
    """
    i = content.find(label)
    if i < 0:
        return ""
    start = i + len(label)

    # Next 'טלפון' label.
    next_phone = content.find(_PHONE_LABEL, start)
    # Next section banner after `start`.
    banner_match = _BANNER_RE.search(content, pos=start)
    next_banner = banner_match.start() if banner_match else -1

    candidates = [start + _PHONE_REGION_CAP]
    if next_phone >= 0:
        candidates.append(next_phone)
    if next_banner >= 0:
        candidates.append(next_banner)
    upper = min(candidates)

    window = content[start:upper]
    for m in re.finditer(r"\d{9,12}", window):
        digits = m.group(0)
        if len(digits) == 9 and _parse_ddmmyyyy(digits):
            continue
        return digits
    return ""


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
    inserts an extra digit inside a digit-box row (e.g. ex2's injury
    date comes out as "120182005" instead of "12082005").
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
    md_fields: dict,
    selected: list[dict],
    health_fund: str,
) -> str:
    lines: list[str] = []
    lines.append("=== FORM 283 SPATIAL EXTRACTION ===")
    lines.append("")
    lines.append("These values are pre-computed from the Azure DI text stream")
    lines.append("(after spaced-digit collapse) and from polygon-based selection")
    lines.append("marks.  They are authoritative for every date / ID / phone /")
    lines.append("checkbox field on Form 283 — use them verbatim and do NOT")
    lines.append("re-derive from the markdown body below.")
    lines.append("")
    lines.append("-- Dates (DDMMYYYY, extracted from the markdown after each label) --")
    lines.append(f"formReceiptDateAtClinic: {_fmt_date(md_fields['receipt_date'])}   [label 'תאריך קבלת הטופס בקופה']")
    lines.append(f"formFillingDate:         {_fmt_date(md_fields['filling_date'])}   [label 'תאריך מילוי הטופס']")
    lines.append(f"dateOfInjury:            {_fmt_date(md_fields['injury_date'])}   [label 'תאריך הפגיעה']")
    lines.append(f"dateOfBirth:             {_fmt_date(md_fields['birth_date'])}   [label 'תאריך לידה']")
    lines.append("")
    lines.append("-- Identifiers & phones --")
    lines.append(f"idNumber:                {md_fields['id_number'] or '(not found)'}   [label 'ת.ז.']")
    lines.append(f"mobilePhone:             {md_fields['mobile_phone'] or '(not found)'}   [label 'טלפון נייד']")
    lines.append(f"landlinePhone:           {md_fields['landline_phone'] or '(not found)'}   [label 'טלפון קווי']")
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
# Polygon helpers (for checkbox extraction only)
# ---------------------------------------------------------------------------

def _poly_center(polygon: list[float]) -> tuple[float, float]:
    xs = polygon[0::2]
    ys = polygon[1::2]
    return sum(xs) / len(xs), sum(ys) / len(ys)


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
# Digit-box collapse (Markdown body)
# ---------------------------------------------------------------------------

# Match 8 to 12 digits with optional separators (spaces, brackets, pipes, dots).
# We use digit-with-optional-separators {7,11} followed by a final digit, so
# the total digit count is 8-12 — enough to cover 8-digit dates, 9-digit
# recovered-date sequences, 9-digit IDs, and 10-digit mobile numbers.
_SPACED_DIGIT_RE = re.compile(r"(?:\d[\s\[\]|.]*){7,11}\d")


def _collapse_spaced_digits(text: str) -> tuple[str, int]:
    count = 0

    def _replace(m: re.Match) -> str:
        nonlocal count
        digits = re.sub(r"\D", "", m.group(0))
        if len(digits) == 8:
            if _parse_ddmmyyyy(digits):
                count += 1
                return digits
            # 8 digits but not a valid date — still collapse (likely an ID
            # or phone fragment); preserve the raw digits.
            count += 1
            return digits
        if len(digits) == 9:
            recovered = _parse_ddmmyyyy(digits)
            if recovered:
                count += 1
                return recovered
            # 9 digits not a date — likely an Israeli ID.  Collapse to the
            # raw string.
            count += 1
            return digits
        if 10 <= len(digits) <= 12:
            # Phone numbers or noisy ID rows — collapse as-is.
            count += 1
            return digits
        return m.group(0)

    return _SPACED_DIGIT_RE.sub(_replace, text), count


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
