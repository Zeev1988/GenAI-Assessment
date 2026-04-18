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
    # Form header — two date clusters at the top-left of the page
    # Digit boxes sit between the label row (y≈1.00) and the
    # יום/חודש/שנה sub-label row (y≈1.43 / 1.48).
    # ------------------------------------------------------------------

    # תאריך מילוי הטופס  (form filling date — written by clinic)
    # sub-labels span x≈0.97–2.32, so boxes are just above them.
    # y1 extended to 1.65 to capture digit content that sits at the lower
    # edge of the pre-printed boxes (which overlap with sub-label row y≈1.43–1.55).
    "filling_date":    (0.75, 0.95, 2.50, 1.65),

    # תאריך קבלת הטופס בקופה  (date clinic received the form)
    # sub-labels span x≈3.03–4.36
    # y1 extended to 1.70 for the same reason as filling_date above.
    "receipt_date":    (2.85, 1.00, 4.55, 1.70),

    # ------------------------------------------------------------------
    # Section 1 — תאריך הפגיעה  (date of injury)
    # Digit boxes between section header (y≈2.38) and sub-labels (y≈2.56)
    # ------------------------------------------------------------------
    "injury_date":     (1.00, 2.20, 5.20, 2.60),

    # ------------------------------------------------------------------
    # Section 2 — פרטי התובע  (claimant personal details)
    # ------------------------------------------------------------------

    # שם משפחה  (last name) — right column, text below the label
    "last_name":       (5.50, 3.15, 7.75, 3.60),

    # שם פרטי  (first name) — middle column, text below the label
    "first_name":      (3.40, 3.15, 5.40, 3.60),

    # ת.ז.  (ID) — digit boxes extend LEFT from the ת.ז. label (x≈3.14)
    # ס״ב checksum label sits at x≈2.88–3.07.  x1 extended to 3.55 to reach
    # any 10th box (e.g. ex3 has a 10-digit ID whose last box sits near x≈3.35).
    # The ת.ז. and ס״ב label text captured in this wider region contains no
    # digits, so _extract_digit_field silently strips it.
    "id_number":       (0.30, 3.00, 3.55, 3.55),

    # gender — handled by _extract_selected_checkboxes, not here

    # תאריך לידה  (date of birth)
    # Label at y≈3.62–3.78; sub-labels at y≈4.09–4.22; boxes between them
    "birth_date":      (2.20, 3.78, 4.95, 4.10),

    # ------------------------------------------------------------------
    # כתובת  (address) — single row with 6 labelled columns
    # Column label row at y≈4.47–4.60; input row just below: y≈4.62–4.90
    # Column boundaries inferred from label centre-x values
    # ------------------------------------------------------------------
    "street":          (6.10, 4.62, 7.75, 4.90),   # רחוב / תא דואר  cx≈6.39
    "house_number":    (4.90, 4.62, 6.10, 4.90),   # מס׳ בית          cx≈5.21
    "entrance":        (4.25, 4.62, 4.90, 4.90),   # כניסה            cx≈4.51
    "apartment":       (3.55, 4.62, 4.25, 4.90),   # דירה             cx≈3.81
    "city":            (1.80, 4.62, 3.55, 4.90),   # יישוב            cx≈2.42
    "postal_code":     (0.30, 4.62, 1.80, 4.90),   # מיקוד            cx≈1.02

    # ------------------------------------------------------------------
    # Phone fields
    # Labels at y≈4.94–5.10; digit boxes in the row below: y≈5.10–5.30
    # Mobile ("נייד") is the LEFT cluster; landline ("קווי") is RIGHT.
    # "ח" area-code separators at y≈5.14–5.22 are INSIDE these regions.
    # ------------------------------------------------------------------
    "mobile_phone":    (0.30, 5.10, 3.55, 5.30),   # left of "טלפון נייד" label
    "landline_phone":  (4.60, 5.10, 7.15, 5.30),   # left of "טלפון קווי" label

    # ------------------------------------------------------------------
    # Section 3 — פרטי התאונה  (accident details)
    # ------------------------------------------------------------------

    # סוג העבודה  (job type) — free text to the LEFT of "כאשר עבדתי ב"
    # "כאשר" starts at x≈4.05; job text fills x≈0.30–3.50, same line y≈6.03
    "job_type":        (0.30, 6.03, 3.50, 6.32),

    # שעת הפגיעה  (time of injury — HH:MM)
    # Between "כאשר עבדתי ב" (ends x≈4.37) and "בשעה" (starts x≈5.43)
    "time_of_injury":  (4.40, 6.03, 5.40, 6.22),

    # מקום התאונה checkboxes — handled by _extract_selected_checkboxes

    # כתובת מקום התאונה  (accident address)
    # Input line is just below the label row (y≈6.60–6.74)
    "accident_address": (0.30, 6.74, 6.55, 7.05),

    # נסיבות הפגיעה / תיאור התאונה  (accident description — may span 2 lines)
    # Just below "נסיבות" label (y≈7.12–7.27)
    "accident_description": (0.30, 7.27, 6.05, 7.65),

    # האיבר שנפגע  (injured body part)
    # Just below "האיבר שנפגע" label (y≈7.58–7.71)
    "injured_body_part": (0.30, 7.71, 6.90, 7.95),

    # ------------------------------------------------------------------
    # Section 4 — הצהרה  (declaration)
    # ------------------------------------------------------------------

    # שם המבקש  (applicant printed name) — to the LEFT of label (x≈6.96)
    "applicant_name":  (3.20, 9.20, 6.90, 9.60),

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
