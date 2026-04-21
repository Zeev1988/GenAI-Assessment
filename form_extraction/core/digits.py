"""Anchor-label digit reader for Form 283's numeric fields.

Every numeric field is printed next to a stable Hebrew anchor label
(e.g. `ת.ז.`, `תאריך הפגיעה`). For each field we locate the anchor,
carve a scan window, run a digit-box regex, reverse if RTL context is
detected, and validate structurally. Returns None when nothing passes
so the caller can keep the LLM's reading.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date

_ID_ANCHORS: tuple[str, ...] = ("ת. ז.", "ת.ז.", "מספר זהות", "מס' זהות")


@dataclass(frozen=True)
class NumericField:
    anchors: tuple[str, ...]
    sizes: tuple[int, ...]
    validate: Callable[[str], bool]
    override: bool = True
    scan_window: int = 300
    min_gap: int = 1  # 1 = separators required; 0 = also allow contiguous digits


def _is_valid_id(s: str) -> bool:
    return len(s) in (9, 10) and s.isdigit()


def _is_valid_ddmmyyyy(s: str) -> bool:
    if len(s) != 8 or not s.isdigit():
        return False
    try:
        year = int(s[4:8])
        if not (1900 <= year <= 2100):
            return False
        date(year, int(s[2:4]), int(s[:2]))
    except ValueError:
        return False
    return True


def _is_valid_mobile(s: str) -> bool:
    return len(s) == 10 and s.isdigit() and s.startswith("05")


def _is_valid_landline(s: str) -> bool:
    return len(s) == 9 and s.isdigit() and s[0] == "0" and s[1] in "234689"


def _is_valid_postal(s: str) -> bool:
    return len(s) in (5, 7) and s.isdigit()


def _is_valid_hhmm(s: str) -> bool:
    if len(s) != 4 or not s.isdigit():
        return False
    return 0 <= int(s[:2]) <= 23 and 0 <= int(s[2:]) <= 59


def _is_valid_short(s: str) -> bool:
    return 1 <= len(s) <= 4 and s.isdigit()


# Override=True: strong check, parser replaces LLM value silently.
# Override=False: weak check, validate.py surfaces a warning on mismatch.
_FIELDS: dict[str, NumericField] = {
    "idNumber": NumericField(_ID_ANCHORS, (10, 9), _is_valid_id),
    "dateOfInjury": NumericField(("תאריך הפגיעה",), (8,), _is_valid_ddmmyyyy),
    "dateOfBirth": NumericField(("תאריך לידה",), (8,), _is_valid_ddmmyyyy),
    "formFillingDate": NumericField(("תאריך מילוי הטופס",), (8,), _is_valid_ddmmyyyy),
    "formReceiptDateAtClinic": NumericField(
        ("תאריך קבלת הטופס בקופה", "תאריך קבלת הטופס במרפאה"), (8,), _is_valid_ddmmyyyy,
    ),
    "mobilePhone": NumericField(("טלפון נייד",), (10,), _is_valid_mobile, scan_window=150),
    "landlinePhone": NumericField(("טלפון קווי",), (9,), _is_valid_landline, scan_window=150),
    "postalCode": NumericField(("מיקוד",), (7, 5), _is_valid_postal, scan_window=80, min_gap=0),
    "timeOfInjury": NumericField(("שעת הפגיעה",), (4,), _is_valid_hhmm, scan_window=80),
    "apartment": NumericField(
        ("דירה",), (4, 3, 2, 1), _is_valid_short, override=False, scan_window=40, min_gap=0,
    ),
    "entrance": NumericField(
        ("כניסה",), (2, 1), _is_valid_short, override=False, scan_window=40, min_gap=0,
    ),
    "houseNumber": NumericField(
        ("מספר בית",), (4, 3, 2, 1), _is_valid_short, override=False, scan_window=60, min_gap=0,
    ),
}


# Used to truncate a scan window at the next known anchor.
_ALL_ANCHORS: tuple[str, ...] = tuple(
    {label for spec in _FIELDS.values() for label in spec.anchors}
)


def _box_run_regex(n_digits: int, min_gap: int = 1, max_gap: int = 6) -> re.Pattern[str]:
    if n_digits <= 1:
        return re.compile(r"\d")
    gap = r"\D{" + str(min_gap) + r"," + str(max_gap) + r"}"
    return re.compile(r"\d(?:" + gap + r"\d){" + str(n_digits - 1) + r"}")


_DIGIT_RE = re.compile(r"\d")
_HEBREW_RE = re.compile(r"[\u0590-\u05ff\ufb1d-\ufb4f]")


def _digits_only(s: str) -> str:
    return "".join(_DIGIT_RE.findall(s))


def _scan_window_after(markdown: str, anchor: str, window_size: int) -> str | None:
    """Text past `anchor`, capped at window_size and cut at the next anchor."""
    idx = markdown.find(anchor)
    if idx < 0:
        return None
    start = idx + len(anchor)
    window = markdown[start : start + window_size]

    next_anchor_pos: int | None = None
    for other in _ALL_ANCHORS:
        if other == anchor:
            continue
        pos = window.find(other)
        if pos < 0:
            continue
        if next_anchor_pos is None or pos < next_anchor_pos:
            next_anchor_pos = pos
    if next_anchor_pos is not None:
        window = window[:next_anchor_pos]
    return window


def _hebrew_precedes_on_same_line(window: str, digit_pos: int) -> bool:
    """True when Hebrew text appears earlier on the line — means digits are in RTL visual order."""
    line_start = window.rfind("\n", 0, digit_pos) + 1
    return bool(_HEBREW_RE.search(window[line_start:digit_pos]))


def _read_box_digits(window: str, regex: re.Pattern[str]) -> tuple[str, int] | None:
    m = regex.search(window)
    if m is None:
        return None
    digits = _digits_only(m.group(0))
    if _hebrew_precedes_on_same_line(window, m.start()):
        digits = digits[::-1]
    return digits, m.start()


def parse_numeric(markdown: str, field: str) -> str | None:
    """Return the validated digit string for `field`, or None."""
    spec = _FIELDS.get(field)
    if spec is None:
        return None
    for anchor in spec.anchors:
        window = _scan_window_after(markdown, anchor, spec.scan_window)
        if window is None:
            continue
        for n in spec.sizes:
            res = _read_box_digits(window, _box_run_regex(n, spec.min_gap))
            if res is None:
                continue
            digits, _ = res
            if len(digits) < n:
                continue
            candidate = digits[:n]
            if spec.validate(candidate):
                return candidate
    return None


def parse_id(markdown: str) -> str | None:
    return parse_numeric(markdown, "idNumber")


def parse_date(markdown: str, field: str) -> tuple[str, str, str] | None:
    """Returns (dd, mm, yyyy) or None."""
    s = parse_numeric(markdown, field)
    if s is None or len(s) != 8:
        return None
    return s[:2], s[2:4], s[4:8]


def override_fields() -> tuple[str, ...]:
    return tuple(name for name, spec in _FIELDS.items() if spec.override)


def warn_only_fields() -> tuple[str, ...]:
    return tuple(name for name, spec in _FIELDS.items() if not spec.override)
