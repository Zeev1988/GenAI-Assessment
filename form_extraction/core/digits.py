"""Anchor-label digit reader for Form 283's numeric fields.

Every numeric field on Form 283 (ID, four dates, two phones, postal
code, time of injury, and the three short address digits) is printed
as a row of boxes or a short digit run preceded by a stable Hebrew
anchor label. OCR Markdown preserves both the anchor and the box
contents, though separators vary: sometimes whitespace-delimited
digits, sometimes ``|``-fenced cells, sometimes contiguous digits.

The parser is intentionally narrow. For each field we:

1. Look up the field's spec in ``_FIELDS`` — anchors, acceptable
   digit counts, a structural validator, a scan-window size, and a
   minimum separator-gap size.
2. Locate an anchor in the Markdown and carve a scan window after it,
   truncated at the next known anchor. The truncation keeps a digit
   row from one field from being read as another field's value.
3. Run a box-pattern regex sized for each acceptable digit count —
   digit, ``\\D{min_gap,6}``, digit, …, repeated. ``min_gap=1`` (the
   default) requires separators between every pair and rejects
   free-text dates like ``14.04.1999``; ``min_gap=0`` also allows
   contiguous digits for fields OCR often emits tightly packed (postal
   codes, short address digits).
4. If the digit run appears on a line that also contains Hebrew
   characters (U+0590–U+05FF or the presentation-forms block), Azure
   DI emitted it in visual right-to-left order; we reverse it back.
5. Validate structurally and return the first candidate that passes.
   Anything that fails yields ``None`` so the caller can fall back to
   the LLM's reading.

The result is trustworthy precisely *because* it is narrow. When the
parser speaks at all, every check has passed. Displaced-anchor cases
(ex2 ``idNumber``, ex3 ``dateOfBirth`` on the sample data) return
``None`` and the LLM's value survives.

Override vs warn-only. Fields whose structural validator is strong
(ID, calendar-valid dates, prefix-anchored phones, 5/7-digit postal,
HH:MM time) have ``override=True``: when the parser reads a value
that passes its check, the extractor replaces the LLM's value
silently. Short fields (apartment, entrance, house number) have
weaker checks — any 1–4 digit run passes — so they use
``override=False``. ``validate.py`` surfaces a warning when the
parser's read differs from the LLM's value, and the LLM's value is
kept; the reviewer decides.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date

# ---------------------------------------------------------------------------
# Anchor tuples
# ---------------------------------------------------------------------------

_ID_ANCHORS: tuple[str, ...] = (
    "ת. ז.",
    "ת.ז.",
    "מספר זהות",
    "מס' זהות",
)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NumericField:
    """Spec for one anchor-parseable numeric field.

    ``anchors``
        Hebrew label(s) that precede the field's digit box. Tried in
        order — first one to yield a validator-passing read wins.
    ``sizes``
        Acceptable digit counts, largest first. ``(10, 9)`` lets an
        ID read 10 when OCR spilled a check digit and 9 when it did
        not.
    ``validate``
        Structural check applied to the digit string before the read
        is accepted.
    ``override``
        When ``True`` the extractor silently replaces the LLM value
        with the parser's value. When ``False`` the parser runs in
        ``validate.py`` instead and a mismatch raises a warning;
        the LLM value is left alone.
    ``scan_window``
        Characters past the anchor to consider. Tight windows cut
        false-positive risk on short fields.
    ``min_gap``
        Minimum non-digit characters between two digits of the
        box run. ``1`` forbids contiguous digits (rejects free-text
        dates like ``14.04.1999``); ``0`` allows them (postal code
        and short address fields).
    """

    anchors: tuple[str, ...]
    sizes: tuple[int, ...]
    validate: Callable[[str], bool]
    override: bool = True
    scan_window: int = 300
    min_gap: int = 1


# ---------------------------------------------------------------------------
# Structural validators
# ---------------------------------------------------------------------------


def _is_valid_id(s: str) -> bool:
    return len(s) in (9, 10) and s.isdigit()


def _is_valid_ddmmyyyy(s: str) -> bool:
    if len(s) != 8 or not s.isdigit():
        return False
    dd, mm, yyyy = s[:2], s[2:4], s[4:8]
    try:
        year = int(yyyy)
        if not (1900 <= year <= 2100):
            return False
        date(year, int(mm), int(dd))
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


# ---------------------------------------------------------------------------
# Field registry — single source of truth for anchor-label extraction
# ---------------------------------------------------------------------------


_FIELDS: dict[str, NumericField] = {
    # Strong structural checks — parser overrides the LLM when it reads.
    "idNumber": NumericField(_ID_ANCHORS, (10, 9), _is_valid_id),
    "dateOfInjury": NumericField(
        ("תאריך הפגיעה",), (8,), _is_valid_ddmmyyyy,
    ),
    "dateOfBirth": NumericField(
        ("תאריך לידה",), (8,), _is_valid_ddmmyyyy,
    ),
    "formFillingDate": NumericField(
        ("תאריך מילוי הטופס",), (8,), _is_valid_ddmmyyyy,
    ),
    "formReceiptDateAtClinic": NumericField(
        ("תאריך קבלת הטופס בקופה", "תאריך קבלת הטופס במרפאה"),
        (8,),
        _is_valid_ddmmyyyy,
    ),
    "mobilePhone": NumericField(
        ("טלפון נייד",), (10,), _is_valid_mobile, scan_window=150,
    ),
    "landlinePhone": NumericField(
        ("טלפון קווי",), (9,), _is_valid_landline, scan_window=150,
    ),
    "postalCode": NumericField(
        ("מיקוד",), (7, 5), _is_valid_postal, scan_window=80, min_gap=0,
    ),
    "timeOfInjury": NumericField(
        ("שעת הפגיעה",), (4,), _is_valid_hhmm, scan_window=80,
    ),
    # Weak structural checks — parser surfaces a warning but does not
    # overwrite. validate.py compares the parser's read to the LLM's
    # value and reports a mismatch; the LLM value stays.
    "apartment": NumericField(
        ("דירה",),
        (4, 3, 2, 1),
        _is_valid_short,
        override=False,
        scan_window=40,
        min_gap=0,
    ),
    "entrance": NumericField(
        ("כניסה",),
        (2, 1),
        _is_valid_short,
        override=False,
        scan_window=40,
        min_gap=0,
    ),
    "houseNumber": NumericField(
        ("מספר בית",),
        (4, 3, 2, 1),
        _is_valid_short,
        override=False,
        scan_window=60,
        min_gap=0,
    ),
}


# Union of every anchor — used to truncate a scan window at the next
# known anchor so one field can never latch onto another's box.
_ALL_ANCHORS: tuple[str, ...] = tuple(
    {label for spec in _FIELDS.values() for label in spec.anchors}
)


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------


def _box_run_regex(n_digits: int, min_gap: int = 1, max_gap: int = 6) -> re.Pattern[str]:
    """A run of ``n_digits`` digits separated by ``min_gap``–``max_gap`` non-digits."""
    if n_digits <= 1:
        return re.compile(r"\d")
    gap = r"\D{" + str(min_gap) + r"," + str(max_gap) + r"}"
    return re.compile(r"\d(?:" + gap + r"\d){" + str(n_digits - 1) + r"}")


_DIGIT_RE = re.compile(r"\d")
_HEBREW_RE = re.compile(r"[\u0590-\u05ff\ufb1d-\ufb4f]")


def _digits_only(s: str) -> str:
    return "".join(_DIGIT_RE.findall(s))


def _scan_window_after(
    markdown: str, anchor: str, window_size: int
) -> str | None:
    """Return the text past ``anchor``, capped at ``window_size`` and cut at the next anchor.

    ``None`` when the anchor is absent. Truncation at the next anchor
    prevents, e.g., a noisy injury-date box from latching onto the
    clean DOB box that follows it.
    """
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
    """True when any Hebrew character appears earlier on the digits' line.

    On a line that begins with a Hebrew run (RTL), digits that follow
    are emitted to the logical stream in visual right-to-left order.
    Detect structurally; the caller reverses.
    """
    line_start = window.rfind("\n", 0, digit_pos) + 1
    return bool(_HEBREW_RE.search(window[line_start:digit_pos]))


def _read_box_digits(
    window: str, regex: re.Pattern[str]
) -> tuple[str, int] | None:
    """Find the first box-pattern run, applying the RTL reversal uniformly."""
    m = regex.search(window)
    if m is None:
        return None
    digits = _digits_only(m.group(0))
    if _hebrew_precedes_on_same_line(window, m.start()):
        digits = digits[::-1]
    return digits, m.start()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def parse_numeric(markdown: str, field: str) -> str | None:
    """Return the validated digit string for ``field``, or ``None``.

    Tries each anchor in registration order, each size in the field's
    size tuple (largest first), and each box read against the field's
    structural validator. Returns on first validator-passing read;
    returns ``None`` if nothing passes anywhere.
    """
    spec = _FIELDS.get(field)
    if spec is None:
        return None
    for anchor in spec.anchors:
        window = _scan_window_after(markdown, anchor, spec.scan_window)
        if window is None:
            continue
        for n in spec.sizes:
            regex = _box_run_regex(n, spec.min_gap)
            res = _read_box_digits(window, regex)
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
    """Back-compat wrapper: returns the 9/10-digit ID or ``None``."""
    return parse_numeric(markdown, "idNumber")


def parse_date(markdown: str, field: str) -> tuple[str, str, str] | None:
    """Back-compat wrapper: returns ``(dd, mm, yyyy)`` or ``None``.

    Dates populate a nested ``DatePart`` on ``ExtractedForm``, so the
    extractor wants the parts split. ``parse_numeric`` returns the
    8-digit string; we split here.
    """
    s = parse_numeric(markdown, field)
    if s is None or len(s) != 8:
        return None
    return s[:2], s[2:4], s[4:8]


def override_fields() -> tuple[str, ...]:
    """Names of fields whose parser result should silently replace the LLM value."""
    return tuple(name for name, spec in _FIELDS.items() if spec.override)


def warn_only_fields() -> tuple[str, ...]:
    """Names of fields whose parser result should surface as a validation warning."""
    return tuple(name for name, spec in _FIELDS.items() if not spec.override)
