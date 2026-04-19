"""Fixed bounding-box regions for every input field on Form 283 (page 1).

Each region is the bounding box of the INPUT AREA (the boxes / underlines
where users write), not the printed label itself. ~0.05" padding is added
on each side.

Coordinate system
-----------------
All values are in INCHES from the TOP-LEFT corner of the page.
A region tuple is (x0, y0, x1, y1).

How the regions were derived
----------------------------
Each region was measured on the blank Form 283 PDF in Adobe Acrobat, with
values read in PDF points and converted at 72 pt/inch. For digit-box fields
(dates, ID, phones) we measured the row of printed boxes directly; for
free-text fields we measured the underline or bounded whitespace below the
field label. A handful of fields needed small padding adjustments after
empirical testing on the three ``283_ex*.pdf`` samples — those are called
out in inline comments below.

Because Form 283 is a fixed-layout government form the field positions are
stable across every printed copy; the regions only need calibration once.

Checkbox labels (the keys of ``CHECKBOXES_PAGE_1``) are imported from
``schemas`` so that the checkbox extraction, the JSON schema enum, the
extractor prompt, and the tests can never drift apart.
"""

from __future__ import annotations

from form_extraction.core.schemas import (
    ACCIDENT_LOCATION_LABELS,
    GENDER_LABELS,
    HEALTH_FUND_LABELS,
)

PAGE_1: dict[str, tuple[float, float, float, float]] = {

    # Form header ──────────────────────────────────────────────────────
    "filling_date":    (0.80, 0.63, 2.59, 1.70),
    "receipt_date":    (2.88, 0.64, 4.66, 1.72),

    # Section 1 — תאריך הפגיעה ─────────────────────────────────────────
    "injury_date":     (3.31, 2.25, 5.06, 2.66),

    # Section 2 — פרטי התובע ──────────────────────────────────────────
    "last_name":       (5.67, 3.02, 7.71, 3.57),

    # first_name: "ס״ב / עי" checksum prefix overlaps this region;
    # filtered by content in ocr.py (_CHECKSUM_TOKENS), not by coordinate.
    "first_name":      (3.46, 3.03, 5.59, 3.56),

    # id_number: digit extraction strips any non-digit label text.
    "id_number":       (0.68, 3.27, 3.41, 3.56),

    "birth_date":      (2.20, 3.76, 4.07, 4.08),

    # Address row ──────────────────────────────────────────────────────
    # street: x0 kept wider (4.80) than the labelled column (x≈5.66) because
    # multi-word street names spill left into the house-number column;
    # pure-digit tokens are stripped in ocr.py.
    "street":          (4.80, 4.48, 7.73, 4.90),
    # house_number: _extract_digit_field strips any spillover street text.
    "house_number":    (4.85, 4.46, 5.58, 4.89),
    "entrance":        (4.23, 4.47, 4.81, 4.90),
    "apartment":       (3.45, 4.47, 4.18, 4.89),
    # city: apartment number sits in the next column (x≥3.45) so it won't
    # bleed in; digit filter in ocr.py kept as an extra safety net.
    "city":            (1.46, 4.46, 3.41, 4.89),
    "postal_code":     (0.66, 4.46, 1.38, 4.87),

    # Phone fields ─────────────────────────────────────────────────────
    "mobile_phone":    (0.66, 4.95, 4.17, 5.35),
    "landline_phone":  (4.23, 4.95, 7.74, 5.35),

    # Section 3 — פרטי התאונה ─────────────────────────────────────────
    # job_type: "כאשר עבדתי ב" label starts at x≈3.55; x1=3.52 keeps it out.
    # "סוג העבודה" label centre is at y≈6.26; y1=6.20 keeps it out.
    "job_type":        (0.68, 5.76, 3.52, 6.20),
    # time_of_injury: between "כאשר עבדתי ב" (ends x≈4.37) and "בשעה" (x≈5.43)
    "time_of_injury":  (4.39, 5.91, 5.39, 6.15),
    # accident_address: x1 extended to 7.77 because Hebrew text is written
    # RTL so the first word sits at the highest x; a narrower x1 cuts it off.
    "accident_address": (0.69, 6.59, 7.77, 6.98),
    # accident_description: text sits ABOVE the "נסיבות הפגיעה" label
    # (y≈7.12–7.27); region kept at y=[6.90, 7.27].
    "accident_description": (0.30, 6.90, 7.75, 7.27),
    "injured_body_part": (0.68, 7.58, 7.75, 7.88),
}


# ---------------------------------------------------------------------------
# Checkbox regions for every tickable box on page 1.
#
# Each entry maps a label (dict key) to the bounding box of its printed
# checkbox square.  _extract_selected_checkboxes checks whether an Azure DI
# selection-mark centre falls inside any box and returns the corresponding
# label directly — no word-proximity heuristic required.
#
# The label strings are re-used from schemas.py so that the extraction
# output is by construction one of the values the JSON schema will accept.
# Two membership-status checkboxes ("הנפגע חבר … / אינו חבר") appear in
# Section 5 alongside the fund-name boxes but do not correspond to any
# schema enum; they are kept here because the OCR still resolves them and
# the log message distinguishes "no fund selected" from "nothing marked
# in Section 5 at all".
# ---------------------------------------------------------------------------

assert GENDER_LABELS == ("זכר", "נקבה"), "Gender label order changed; update regions"
assert set(ACCIDENT_LOCATION_LABELS) == {
    "במפעל",
    "ת. דרכים בעבודה",
    "ת. דרכים בדרך לעבודה/מהעבודה",
    "תאונה בדרך ללא רכב",
    "אחר",
}, "Accident-location labels changed; update regions"
assert set(HEALTH_FUND_LABELS) == {
    "כללית",
    "מכבי",
    "מאוחדת",
    "לאומית",
}, "Health-fund labels changed; update regions"


CHECKBOXES_PAGE_1: dict[str, tuple[float, float, float, float]] = {

    # Section 2 — gender
    GENDER_LABELS[0]: (6.90, 3.85, 7.25, 4.12),   # "זכר" (mark observed at (7.08, 3.98))
    GENDER_LABELS[1]: (6.40, 3.85, 6.75, 4.12),   # "נקבה"

    # Section 3 — accident location
    ACCIDENT_LOCATION_LABELS[0]: (6.667, 6.351, 6.907, 6.515),  # "במפעל"
    ACCIDENT_LOCATION_LABELS[1]: (6.023, 6.351, 6.275, 6.540),  # "ת. דרכים בעבודה"
    ACCIDENT_LOCATION_LABELS[2]: (4.886, 6.351, 5.126, 6.528),  # "ת. דרכים בדרך לעבודה/מהעבודה"
    ACCIDENT_LOCATION_LABELS[3]: (2.917, 6.364, 3.157, 6.528),  # "תאונה בדרך ללא רכב"
    ACCIDENT_LOCATION_LABELS[4]: (1.566, 6.351, 1.806, 6.528),  # "אחר"

    # Section 5 — health-fund membership status (not a schema enum; kept for log clarity)
    "הנפגע חבר בקופת חולים":       (7.247, 9.811, 7.487, 9.962),
    "הנפגע אינו חבר בקופת חולים":  (7.273, 10.025, 7.475, 10.227),

    # Section 5 — health-fund name
    HEALTH_FUND_LABELS[0]: (5.669, 9.811, 5.884, 9.975),  # "כללית"
    HEALTH_FUND_LABELS[2]: (5.025, 9.785, 5.265, 9.962),  # "מאוחדת"
    HEALTH_FUND_LABELS[1]: (4.306, 9.785, 4.533, 9.962),  # "מכבי"
    HEALTH_FUND_LABELS[3]: (3.725, 9.811, 3.965, 9.975),  # "לאומית"
}
