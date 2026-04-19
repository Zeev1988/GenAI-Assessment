"""Fixed bounding-box regions for every input field on Form 283 (page 1).

Coordinates are derived from calibration_words.json — the Azure DI output
of the blank (empty) form run through prebuilt-layout.  Each region is the
bounding box of the INPUT AREA (the boxes / underlines where users write),
not the label itself.  ~0.05" padding has been added on each side.

Coordinate system
-----------------
All values are in INCHES from the TOP-LEFT corner of the page.
A region tuple is (x0, y0, x1, y1).

How the regions were derived
-----------------------------
For each field we locate the surrounding LABEL words in calibration_words.json,
then infer the input box location:

  • Digit-box fields (dates, ID, phones): the boxes sit ABOVE the
    יום/חודש/שנה sub-labels or to the LEFT of the field label (RTL).
  • Text fields (names, addresses, free text): the input area is
    directly below the column header label.

Reference anchors from calibration (blank form, page 1)
--------------------------------------------------------
Header area
  "תאריך מילוי הטופס" label    y≈0.84–1.00
  filling-date sub-labels       y≈1.43–1.55  x≈0.97–2.32
  "תאריך קבלת הטופס בקופה"      y≈0.88–1.06
  receipt-date sub-labels       y≈1.48–1.60  x≈3.03–4.36

Section 1
  "תאריך הפגיעה" label          y≈2.22–2.38
  injury-date sub-labels        y≈2.56–2.68  x≈3.51–4.75

Section 2
  "שם משפחה" label              y≈3.03–3.15  x≈6.80–7.45
  "שם פרטי" label               y≈3.03–3.15  x≈4.99–5.48
  "ת.ז." label                  y≈3.03–3.15  x≈3.14–3.35
  "ס״ב" (checksum)              y≈3.17–3.28  x≈2.88–3.07
  "תאריך לידה" label            y≈3.62–3.78  x≈4.90–5.55
  birth-date sub-labels         y≈4.09–4.22  x≈2.44–3.69
  address column labels row     y≈4.47–4.60
    "רחוב/תא דואר" cx≈6.39      "מס׳ בית" cx≈5.21
    "כניסה" cx≈4.51             "דירה" cx≈3.81
    "יישוב" cx≈2.42             "מיקוד" cx≈1.02
  "טלפון קווי" label            y≈4.95–5.10  x≈7.17–7.69
  "טלפון נייד" label            y≈4.94–5.10  x≈3.59–4.12
  phone "ח" separators          y≈5.14–5.22  x≈1.41 (mobile), x≈4.98 (landline)

Section 3
  "כאשר עבדתי ב"                y≈6.03–6.16  x≈3.55–4.37
  "בשעה"                        y≈6.03–6.16  x≈5.43–5.78
  "בתאריך"                      y≈6.03–6.16  x≈7.25–7.69
  "סוג העבודה"                  y≈6.19–6.32  x≈1.93–2.58
  "כתובת מקום התאונה" label      y≈6.60–6.74
  "נסיבות הפגיעה / תאור התאונה" y≈7.12–7.27
  "האיבר שנפגע" label           y≈7.58–7.71

Section 4
  "חתימה" + "X" marker          y≈9.09–9.26  x≈2.52–3.08
  "שם המבקש" label              y≈9.13–9.28  x≈6.96–7.69

Section 5
  "כללית/מאוחדת/מכבי/לאומית"   y≈9.84–9.98  (checkboxes — not here)
  "מהות התאונה (אבחנות רפואיות)" y≈10.29–10.46
"""

from __future__ import annotations

PAGE_1: dict[str, tuple[float, float, float, float]] = {

    # ------------------------------------------------------------------
    # Coordinates derived from user-measured PDF annotation (Book2.xlsx).
    # Original values are in PDF points; converted to inches at 72 pt/inch.
    # Fields not present in the spreadsheet retain manually calibrated values.
    # ------------------------------------------------------------------

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

    # gender — handled by _extract_selected_checkboxes, not here

    "birth_date":      (2.20, 3.76, 4.07, 4.08),

    # Address row ──────────────────────────────────────────────────────
    # street: x0 kept at 4.80 (wider than the labelled column x≈5.66)
    # because multi-word street names often spill left into the house-number
    # column; pure-digit tokens are stripped in ocr.py.
    "street":          (4.80, 4.48, 7.73, 4.90),
    # house_number: _extract_digit_field strips any spillover street text.
    "house_number":    (4.85, 4.46, 5.58, 4.89),
    "entrance":        (4.23, 4.47, 4.81, 4.90),
    "apartment":       (3.45, 4.47, 4.18, 4.89),
    # city: apartment number is in the next column (x≥3.45) so won't bleed
    # in; digit filter in ocr.py kept as an extra safety net.
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

    # מקום התאונה checkboxes — handled by _extract_selected_checkboxes

    # accident_address: x1 extended to 7.77 (user-measured) — critical fix.
    # Hebrew text is written RTL so the first word ('לוונברג') sits at the
    # highest x; old x1=6.55 cut it off entirely.
    "accident_address": (0.69, 6.59, 7.77, 6.98),

    # accident_description: text sits ABOVE the "נסיבות הפגיעה" label
    # (y≈7.12–7.27); region kept at y=[6.90, 7.27] which is confirmed
    # working. x1=7.75 per user measurement.
    "accident_description": (0.30, 6.90, 7.75, 7.27),

    # injured_body_part: user-measured region.
    "injured_body_part": (0.68, 7.58, 7.75, 7.88),

    # ------------------------------------------------------------------
    # Section 4 — הצהרה  (declaration)
    # ------------------------------------------------------------------

    # שם המבקש  (applicant printed name) — to the LEFT of label (x≈6.96)
    # y0 raised 9.20 → 8.85, y1 raised 9.60 → 9.20: the printed name appears
    # ABOVE the "שם המבקש" label (y≈9.13–9.28); old region was below the label.
    "applicant_name":  (3.20, 8.85, 6.90, 9.20),

    # חתימה  (signature) — to the LEFT of the "חתימה" label (x≈2.66)
    "signature":       (0.30, 9.09, 2.50, 9.35),

    # ------------------------------------------------------------------
    # Section 5 — למילוי ע"י המוסד הרפואי  (filled by medical institution)
    # ------------------------------------------------------------------

    # health-fund checkboxes — handled by _extract_selected_checkboxes

    # מהות התאונה + אבחנות רפואיות
    # "מהות התאונה (אבחנות רפואיות):" label ends at y≈10.46
    # Input area is below the label, before the page footer (y≈10.75)
    "nature_of_accident": (0.30, 10.29, 5.35, 10.50),   # same row as label, left side
    "medical_diagnoses":  (0.30, 10.50, 7.75, 10.80),   # below label row
}

# ---------------------------------------------------------------------------
# Checkbox regions for every tickable box on page 1.
#
# Each entry maps a known label to the bounding box of its printed checkbox
# square (x0, y0, x1, y1) in inches.  _extract_selected_checkboxes checks
# whether an Azure DI selection-mark centre falls inside any box and returns
# the corresponding label directly — no word-proximity heuristic required.
#
# Source: user-measured PDF coordinates converted at 72 pt/inch.
# Gender boxes are estimated from observed Azure DI mark positions because
# they were not included in the measurement spreadsheet.
# ---------------------------------------------------------------------------

CHECKBOXES_PAGE_1: dict[str, tuple[float, float, float, float]] = {

    # Section 2 — gender (estimated; no spreadsheet measurement available)
    "זכר":   (6.90, 3.85, 7.25, 4.12),   # mark observed at (7.08, 3.98)
    "נקבה":  (6.40, 3.85, 6.75, 4.12),   # estimated; no test case available

    # Section 3 — accident location
    "במפעל":                          (6.667, 6.351, 6.907, 6.515),
    "ת. דרכים בעבודה":               (6.023, 6.351, 6.275, 6.540),
    "ת. דרכים בדרך לעבודה/מהעבודה": (4.886, 6.351, 5.126, 6.528),
    "תאונה בדרך ללא רכב":            (2.917, 6.364, 3.157, 6.528),
    "אחר":                            (1.566, 6.351, 1.806, 6.528),

    # Section 5 — health-fund membership status
    "הנפגע חבר בקופת חולים":       (7.247, 9.811, 7.487, 9.962),
    "הנפגע אינו חבר בקופת חולים":  (7.273, 10.025, 7.475, 10.227),

    # Section 5 — health-fund name
    "כללית":   (5.669, 9.811, 5.884, 9.975),
    "מאוחדת":  (5.025, 9.785, 5.265, 9.962),
    "מכבי":    (4.306, 9.785, 4.533, 9.962),
    "לאומית":  (3.725, 9.811, 3.965, 9.975),
}
