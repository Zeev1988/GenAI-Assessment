"""Run Azure Document Intelligence (prebuilt-layout) on an uploaded file.

The OCR stage produces a single ``=== FORM 283 SPATIAL EXTRACTION ===``
document containing pre-computed key/value pairs for every field on Form 283.

Two Azure DI data types are used — both coordinate-based:

  * ``pages[].words[]`` — each word carries a polygon (bounding box in
    inches).  For every text field we collect all words whose centre falls
    inside a pre-calibrated region for that field (see ``field_regions.py``).
    Words are then sorted into reading order: top-to-bottom across rows,
    right-to-left within each Hebrew row.

  * ``pages[].selection_marks[]`` — each checkbox carries a polygon and a
    ``state`` attribute (selected / unselected).  We resolve the label of
    every selected mark by checking whether its centre falls inside a
    pre-calibrated bounding box for a known checkbox (see ``field_regions.py``,
    ``CHECKBOXES_PAGE_1``).

Because Form 283 is a fixed-layout government form the field positions are
stable across every printed copy; the regions need to be calibrated only
once against the blank form (see ``calibrate.py``).

Design rationale
----------------
A single coordinate-based method covers the entire form — no label-anchored
regex, no Markdown stream parsing, no RTL fallback heuristics, no LLM needed
for field location.  The LLM's only remaining job is to copy the structured
header into a typed JSON schema and split DDMMYYYY dates into parts.
"""

from __future__ import annotations

import logging
import re
import time

from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import AnalyzeDocumentRequest
from azure.core.credentials import AzureKeyCredential

from form_extraction.core.config import Settings, get_settings
from form_extraction.core.field_regions import CHECKBOXES_PAGE_1, PAGE_1

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
        )
        result = poller.result()

    elapsed_ms = int((time.perf_counter() - t0) * 1000)
    if not result.pages:
        log.warning("ocr.empty elapsed_ms=%d", elapsed_ms)
        raise RuntimeError("OCR returned no pages; the document may be blank or unreadable.")

    # --- Extract text fields from Azure DI word coordinates ------------------
    coord_fields = _extract_coordinate_fields(result)

    # --- Extract checkboxes from Azure DI polygon selection marks ------------
    selected = _extract_selected_checkboxes(result)
    health_fund = _resolve_health_fund(selected)

    # --- Render the spatial extraction output --------------------------------
    output = _render_spatial_header(coord_fields, selected, health_fund)

    log.info(
        "ocr.done chars=%d elapsed_ms=%d  receipt=%r filling=%r injury=%r "
        "birth=%r id=%r mobile=%r landline=%r fund=%r name=%r/%r",
        len(output), elapsed_ms,
        coord_fields["receipt_date"], coord_fields["filling_date"],
        coord_fields["injury_date"],  coord_fields["birth_date"],
        coord_fields["id_number"],    coord_fields["mobile_phone"],
        coord_fields["landline_phone"], health_fund,
        coord_fields.get("first_name"), coord_fields.get("last_name"),
    )
    return output


# ---------------------------------------------------------------------------
# Polygon helpers (used by both text-field and checkbox extraction)
# ---------------------------------------------------------------------------

def _poly_center(polygon: list[float]) -> tuple[float, float]:
    xs = polygon[0::2]
    ys = polygon[1::2]
    return sum(xs) / len(xs), sum(ys) / len(ys)


# ---------------------------------------------------------------------------
# Coordinate-based field extraction
# ---------------------------------------------------------------------------

_ROW_TOLERANCE = 0.12   # inches — words within this vertical distance are on the same row

# Hebrew tokens that appear in the שם פרטי (first-name) bounding box but are
# NOT part of the name itself.  "עי" / "ס״ב" are the checksum-digit prefix
# labels printed just to the left of the ID-number digit boxes; their centre
# falls inside the first-name column on some copies of the form.
_CHECKSUM_TOKENS: frozenset[str] = frozenset({"עי", "ס״ב"})

# Printed column-header / field-label tokens stamped on the blank form.
# Their centres fall inside the same bounding-box region as the user-written
# content, so they must be stripped after spatial extraction.
# NOTE: "בית" (from "מס׳ בית") rarely appears in Israeli street names as a
# standalone token — acceptable trade-off for this fixed-layout government form.
_FIELD_LABEL_TOKENS: dict[str, frozenset[str]] = {
    "last_name":            frozenset({"שם", "משפחה"}),
    "first_name":           frozenset({"שם", "פרטי"}),
    # "רחוב / תא דואר" + "מס׳ בית" column headers
    "street":               frozenset({"רחוב", "/", "תא", "דואר", "מס׳", "בית"}),
    "entrance":             frozenset({"כניסה"}),
    "apartment":            frozenset({"דירה"}),
    "city":                 frozenset({"יישוב"}),
    # "כתובת מקום התאונה" field label
    "accident_address":     frozenset({"כתובת", "מקום", "התאונה"}),
    # "נסיבות הפגיעה / תאור התאונה" field label
    "accident_description": frozenset({"נסיבות", "הפגיעה", "/", "תאור", "התאונה"}),
    # "האיבר שנפגע" field label
    "injured_body_part":    frozenset({"האיבר", "שנפגע"}),
}


def _strip_label_tokens(text: str, field: str) -> str:
    """Remove printed form-label words from *text* for the given *field*.

    The blank Form 283 has column-header labels whose word centres fall inside
    the same bounding-box region as the user-written content.  These known
    label words are stripped here so they never reach the LLM.
    """
    tokens = _FIELD_LABEL_TOKENS.get(field)
    if not tokens:
        return text
    return " ".join(t for t in text.split() if t not in tokens).strip()


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
    """Extract every field on page 1 using bounding-box regions.

    Dates, ID, and phones use digit-only extraction; free-text fields use
    RTL-aware text extraction with form-label tokens stripped.
    Checkboxes are handled separately by ``_extract_selected_checkboxes``.
    """
    if not result.pages:
        log.warning("ocr.coord.no_pages")
        return {
            "receipt_date": "", "filling_date": "", "injury_date": "", "birth_date": "",
            "id_number": "", "mobile_phone": "", "landline_phone": "", "postal_code": "",
            "last_name": "", "first_name": "",
            "street": "", "house_number": "", "entrance": "", "apartment": "", "city": "",
            "job_type": "", "time_of_injury": "",
            "accident_address": "", "accident_description": "", "injured_body_part": "",
            "applicant_name": "",
        }

    page = result.pages[0]  # all form fields are on page 1

    def _txt(key: str, rtl: bool = True) -> str:
        return _extract_field_text(page, PAGE_1[key], rtl=rtl)

    def _stripped(key: str, rtl: bool = True) -> str:
        return _strip_label_tokens(_txt(key, rtl=rtl), key)

    fields = {
        # Dates (DDMMYYYY)
        "receipt_date":   _extract_date_field(page, PAGE_1["receipt_date"]),
        "filling_date":   _extract_date_field(page, PAGE_1["filling_date"]),
        "injury_date":    _extract_date_field(page, PAGE_1["injury_date"]),
        "birth_date":     _extract_date_field(page, PAGE_1["birth_date"]),
        # Numeric identifiers & phones
        "id_number":      _extract_digit_field(page, PAGE_1["id_number"]),
        "mobile_phone":   _extract_digit_field(page, PAGE_1["mobile_phone"]),
        "landline_phone": _extract_digit_field(page, PAGE_1["landline_phone"]),
        "postal_code":    _extract_digit_field(page, PAGE_1["postal_code"]),
        # Names — strip column-header label and checksum-prefix tokens
        "last_name":  _stripped("last_name"),
        "first_name": _strip_label_tokens(
            " ".join(t for t in _txt("first_name").split() if t not in _CHECKSUM_TOKENS),
            "first_name",
        ),
        # Address columns — street region is intentionally wide to catch long
        # names; pure-digit tokens (house number bleed) are stripped before
        # the column-header label strip.
        "street":      _strip_label_tokens(
            " ".join(t for t in _txt("street").split() if not t.isdigit()), "street"
        ),
        "house_number": _extract_digit_field(page, PAGE_1["house_number"]),
        "entrance":    _stripped("entrance", rtl=False),
        "apartment":   _stripped("apartment", rtl=False),
        "city":        _strip_label_tokens(
            " ".join(t for t in _txt("city").split() if not t.isdigit()), "city"
        ),
        # Accident details
        "job_type":             _txt("job_type"),
        "time_of_injury":       _txt("time_of_injury", rtl=False),
        "accident_address":     _stripped("accident_address"),
        "accident_description": _stripped("accident_description"),
        "injured_body_part":    _stripped("injured_body_part"),
        # Declaration
        "applicant_name": _txt("applicant_name"),
    }

    log.info(
        "ocr.coord.fields receipt=%r filling=%r injury=%r birth=%r id=%r "
        "mobile=%r landline=%r name=%r/%r city=%r job=%r",
        fields["receipt_date"], fields["filling_date"], fields["injury_date"],
        fields["birth_date"], fields["id_number"],
        fields["mobile_phone"], fields["landline_phone"],
        fields["first_name"], fields["last_name"],
        fields["city"], fields["job_type"],
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
    def _val(key: str) -> str:
        return coord_fields.get(key) or "(not found)"

    def _opt(key: str) -> str:
        """Optional field — blank is fine, don't say '(not found)'."""
        return coord_fields.get(key) or ""

    lines: list[str] = []
    lines.append("=== FORM 283 SPATIAL EXTRACTION ===")
    lines.append("")
    lines.append("These values are pre-computed from Azure DI word coordinates.")
    lines.append("Each field was extracted by collecting all words whose centre")
    lines.append("falls inside a calibrated bounding-box region for that field.")
    lines.append("Checkboxes are resolved from polygon-based selection marks.")
    lines.append("They are authoritative — use them verbatim and do NOT")
    lines.append("re-derive from the form body.")
    lines.append("")
    lines.append("-- Dates (DDMMYYYY, extracted from fixed coordinate regions) --")
    lines.append(f"formReceiptDateAtClinic: {_fmt_date(coord_fields['receipt_date'])}")
    lines.append(f"formFillingDate:         {_fmt_date(coord_fields['filling_date'])}")
    lines.append(f"dateOfInjury:            {_fmt_date(coord_fields['injury_date'])}")
    lines.append(f"dateOfBirth:             {_fmt_date(coord_fields['birth_date'])}")
    lines.append("")
    lines.append("-- Identifiers & phones --")
    lines.append(f"idNumber:                {_val('id_number')}")
    lines.append(f"mobilePhone:             {_val('mobile_phone')}")
    lines.append(f"landlinePhone:           {_val('landline_phone')}")
    lines.append("")
    lines.append("-- Selected checkboxes (from Azure DI selection marks + coordinate region lookup) --")
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
    lines.append("-- Free-text fields (extracted from fixed coordinate regions) --")
    lines.append(f"lastName:            {_val('last_name')}")
    lines.append(f"firstName:           {_val('first_name')}")
    lines.append(f"street:              {_val('street')}")
    lines.append(f"houseNumber:         {_val('house_number')}")
    lines.append(f"entrance:            {_opt('entrance')}")
    lines.append(f"apartment:           {_opt('apartment')}")
    lines.append(f"city:                {_val('city')}")
    lines.append(f"postalCode:          {_opt('postal_code')}")
    lines.append(f"jobType:             {_opt('job_type')}")
    lines.append(f"timeOfInjury:        {_opt('time_of_injury')}")
    lines.append(f"accidentAddress:     {_opt('accident_address')}")
    lines.append(f"accidentDescription: {_opt('accident_description')}")
    lines.append(f"injuredBodyPart:     {_opt('injured_body_part')}")
    lines.append(f"applicantName:       {_opt('applicant_name')}")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Checkbox extraction — coordinate-based
# ---------------------------------------------------------------------------

_FUND_NAMES: frozenset[str] = frozenset({"כללית", "מכבי", "מאוחדת", "לאומית"})


def _extract_selected_checkboxes(result) -> list[dict]:
    """Return one dict per selected checkbox, with label looked up from CHECKBOXES_PAGE_1.

    For each Azure DI selection mark whose state is 'selected', we check
    whether its centre falls inside one of the calibrated checkbox regions.
    If it does, the hardcoded label is returned directly.  Marks that fall
    outside every known region are silently ignored (they are likely
    stray marks or form artefacts).
    """
    out: list[dict] = []
    if not result.pages:
        return out

    for mark in (result.pages[0].selection_marks or []):
        if (getattr(mark, "state", "") or "").lower() != "selected":
            continue
        polygon = getattr(mark, "polygon", None)
        if not polygon or len(polygon) < 4:
            continue

        mcx, mcy = _poly_center(polygon)
        label = None
        for name, (x0, y0, x1, y1) in CHECKBOXES_PAGE_1.items():
            if x0 <= mcx <= x1 and y0 <= mcy <= y1:
                label = name
                break

        if label is None:
            log.debug("ocr.checkbox.unmatched mark=(%.2f,%.2f)", mcx, mcy)
            continue

        log.info("ocr.checkbox.sel mark=(%.2f,%.2f) → label=%r", mcx, mcy, label)
        out.append({"x": mcx, "y": mcy, "label": label})

    return out


def _resolve_health_fund(selected: list[dict]) -> str:
    for mark in selected:
        if mark["label"] in _FUND_NAMES:
            return mark["label"]
    return ""
