"""Anchor-label digit parser: dates, IDs, and other numeric fields read from OCR Markdown."""

from __future__ import annotations

from form_extraction.core.digits import (
    override_fields,
    parse_date,
    parse_id,
    parse_numeric,
    warn_only_fields,
)

# ---------------------------------------------------------------------------
# parse_date
# ---------------------------------------------------------------------------


def test_parse_date_reads_clean_8_digit_box() -> None:
    md = "תאריך הפגיעה\n\n[1]6 0| 4 2 | 0 | 2 | 2|\nשנה חודש יום"
    assert parse_date(md, "dateOfInjury") == ("16", "04", "2022")


def test_parse_date_handles_varied_separators() -> None:
    md = "תאריך מילוי הטופס\n[2|5 0 1 2 0 2 3\nשנה חודש יום"
    assert parse_date(md, "formFillingDate") == ("25", "01", "2023")


def test_parse_date_skips_stray_leading_digit() -> None:
    # ex2-style noise: a stray digit jammed into the middle of the box.
    # Calendar validation rules out shift=0 ("12018200"→year 8200) and
    # shift=1 ("20182005"→month 18); in that case we return None.
    md = "תאריך הפגיעה\n\n[1]2 01 8 2 0 0 5\nשנה חודש יום"
    assert parse_date(md, "dateOfInjury") is None


def test_parse_date_returns_none_for_invalid_calendar() -> None:
    # "31/02/2020" is invalid (February doesn't have 31 days).
    md = "תאריך הפגיעה\n3 1 0 2 2 0 2 0"
    assert parse_date(md, "dateOfInjury") is None


def test_parse_date_returns_none_for_out_of_range_year() -> None:
    md = "תאריך הפגיעה\n1 5 0 4 3 0 0 0"
    assert parse_date(md, "dateOfInjury") is None


def test_parse_date_returns_none_when_anchor_missing() -> None:
    md = "some unrelated markdown with no anchor here"
    assert parse_date(md, "dateOfInjury") is None


def test_parse_date_returns_none_when_too_few_digits() -> None:
    md = "תאריך הפגיעה\n1 5 0 4\n"
    assert parse_date(md, "dateOfInjury") is None


def test_parse_date_uses_fallback_anchor() -> None:
    md = "תאריך קבלת הטופס במרפאה\n0 2 0 2 1 9 9 9\nשנה חודש יום"
    assert parse_date(md, "formReceiptDateAtClinic") == ("02", "02", "1999")


def test_parse_date_scan_window_does_not_cross_fields() -> None:
    # Place anchor, then padding > scan window, then digits. Parser
    # should not reach the digits.
    md = "תאריך הפגיעה" + ("\n" * 350) + "0 1 0 1 2 0 2 0"
    assert parse_date(md, "dateOfInjury") is None


def test_parse_date_stops_at_next_anchor() -> None:
    # When its own box is noisy (rejected by box-pattern) and a
    # neighbouring field's clean box follows, the scan must stop at the
    # next anchor instead of latching onto the wrong field.
    md = (
        "תאריך הפגיעה\n[1]2 01 8 2 0 0 5\n"    # noisy injury box → no match
        "תאריך לידה\n[1 ] 4 1| 0 1 9 9 0\n"     # clean DOB box, but wrong field
    )
    assert parse_date(md, "dateOfInjury") is None
    # The DOB anchor reaches its own clean box just fine.
    assert parse_date(md, "dateOfBirth") == ("14", "10", "1990")


# ---------------------------------------------------------------------------
# parse_id
# ---------------------------------------------------------------------------


def test_parse_id_reads_10_digit_block() -> None:
    md = "ת. ז.\nס״ב\n8 | 7 |7 |5 | 2 | 4 5 | 6 3 1"
    assert parse_id(md) == "8775245631"


def test_parse_id_reads_9_digit_block() -> None:
    md = "ת.ז.\n0 2 2 4 5 6 1 2 0"
    assert parse_id(md) == "022456120"


def test_parse_id_accepts_alternative_anchor_label() -> None:
    md = "מספר זהות\n1 2 3 4 5 6 7 8 9"
    assert parse_id(md) == "123456789"


def test_parse_id_returns_none_when_anchor_missing() -> None:
    md = "no id on this page"
    assert parse_id(md) is None


def test_parse_id_returns_none_when_no_digits_after_anchor() -> None:
    # ex2-style: anchor present, digits displaced to another part of
    # the document beyond the scan window.
    md = "ת.ז.\nס״ב\n\nהלוי\n\n" + ("\n" * 400) + "0 2 2 4 5 6 1 2 0"
    assert parse_id(md) is None


def test_parse_id_truncates_over_long_digit_runs() -> None:
    # When OCR swallows an adjacent field, we take the first 10.
    md = "ת.ז.\n1 2 3 4 5 6 7 8 9 0 1 2 3 4 5"
    assert parse_id(md) == "1234567890"


def test_parse_id_reverses_when_hebrew_shares_line() -> None:
    # ex3-style: Hebrew letters on the same line as the digit box
    # flip the digits into visual-RTL order. The parser reverses them
    # back to reading order.
    md = "ת. ז.\nס״ב\nעי 7 6 5 1| 2 5 | 4 3 3 | 0"
    assert parse_id(md) == "0334521567"


def test_parse_id_does_not_reverse_when_line_is_digits_only() -> None:
    # ex1-style: the digit box lives on its own line, Hebrew lives on
    # the previous line — no RTL context on the digits' line.
    md = "ת. ז.\nס״ב\n8 | 7 |7 |5 | 2 | 4 5 | 6 3 1"
    assert parse_id(md) == "8775245631"


def test_parse_date_reverses_when_hebrew_shares_line() -> None:
    # A hypothetical RTL-context date box: Hebrew letters on the same
    # line cause the digits to be emitted in reverse visual order.
    # 9991/40/41 reversed = 14/04/1999.
    md = "תאריך הפגיעה\nקוד 9 9 9 1 4 0 4 1"
    assert parse_date(md, "dateOfInjury") == ("14", "04", "1999")


# ---------------------------------------------------------------------------
# parse_numeric — phones
# ---------------------------------------------------------------------------


def test_parse_numeric_mobile_reads_clean_10_digit_box() -> None:
    md = "טלפון נייד\n0 5 4 | 1 2 3 | 4 5 6 7"
    assert parse_numeric(md, "mobilePhone") == "0541234567"


def test_parse_numeric_mobile_rejects_non_prefix() -> None:
    # 10 digits but doesn't start with 05 — structural check refuses.
    md = "טלפון נייד\n6 1 9 7 6 5 6 0 5 4"
    assert parse_numeric(md, "mobilePhone") is None


def test_parse_numeric_landline_reads_clean_9_digit_box() -> None:
    md = "טלפון קווי\n0 3 | 1 2 3 | 4 5 6 7"
    assert parse_numeric(md, "landlinePhone") == "031234567"


def test_parse_numeric_landline_rejects_wrong_area_code() -> None:
    # 9 digits starting with 01 — not a valid Israeli landline prefix.
    md = "טלפון קווי\n0 1 2 3 4 5 6 7 8"
    assert parse_numeric(md, "landlinePhone") is None


def test_parse_numeric_phone_reverses_when_hebrew_shares_line() -> None:
    # Hebrew on the digits' line → visual-RTL → reverse to logical.
    # Original visual "7 6 5 4 3 2 1 0 5 0" reversed → "0501234567" (valid mobile).
    md = "טלפון נייד\nקוד 7 6 5 4 3 2 1 0 5 0"
    assert parse_numeric(md, "mobilePhone") == "0501234567"


def test_parse_numeric_phone_returns_none_when_anchor_missing() -> None:
    assert parse_numeric("other content only", "mobilePhone") is None


# ---------------------------------------------------------------------------
# parse_numeric — postal code
# ---------------------------------------------------------------------------


def test_parse_numeric_postal_reads_7_digit_contiguous() -> None:
    # Postal boxes are often emitted contiguously (min_gap=0).
    md = "מיקוד\n1234567"
    assert parse_numeric(md, "postalCode") == "1234567"


def test_parse_numeric_postal_reads_5_digit_when_7_missing() -> None:
    md = "מיקוד\n61234"
    assert parse_numeric(md, "postalCode") == "61234"


def test_parse_numeric_postal_prefers_7_over_5() -> None:
    # With both possible lengths present, the registry tries 7 first.
    md = "מיקוד\n1 2 3 4 5 6 7"
    assert parse_numeric(md, "postalCode") == "1234567"


# ---------------------------------------------------------------------------
# parse_numeric — time of injury
# ---------------------------------------------------------------------------


def test_parse_numeric_time_reads_4_digit_box() -> None:
    md = "שעת הפגיעה\n2 1 | 4 5\nדקות שעות"
    assert parse_numeric(md, "timeOfInjury") == "2145"


def test_parse_numeric_time_rejects_out_of_range() -> None:
    # Hour 25 is invalid.
    md = "שעת הפגיעה\n2 5 0 0"
    assert parse_numeric(md, "timeOfInjury") is None


def test_parse_numeric_time_rejects_minute_out_of_range() -> None:
    md = "שעת הפגיעה\n1 2 7 5"
    assert parse_numeric(md, "timeOfInjury") is None


# ---------------------------------------------------------------------------
# parse_numeric — short address fields (warn-only path)
# ---------------------------------------------------------------------------


def test_parse_numeric_apartment_reads_single_digit() -> None:
    md = "דירה 5\nישוב"
    assert parse_numeric(md, "apartment") == "5"


def test_parse_numeric_apartment_reads_two_digit_contiguous() -> None:
    md = "דירה 12\nישוב"
    assert parse_numeric(md, "apartment") == "12"


def test_parse_numeric_entrance_reads_single_digit() -> None:
    md = "כניסה 3 דירה 1"
    assert parse_numeric(md, "entrance") == "3"


def test_parse_numeric_house_number_stops_at_next_anchor() -> None:
    # Next-anchor truncation keeps the postal box out of the houseNumber scan.
    md = "מספר בית 12\nמיקוד 61234"
    assert parse_numeric(md, "houseNumber") == "12"


# ---------------------------------------------------------------------------
# Registry introspection
# ---------------------------------------------------------------------------


def test_override_fields_include_id_dates_phones_postal_time() -> None:
    names = set(override_fields())
    assert "idNumber" in names
    assert "dateOfInjury" in names
    assert "mobilePhone" in names
    assert "landlinePhone" in names
    assert "postalCode" in names
    assert "timeOfInjury" in names


def test_warn_only_fields_are_short_address_fields() -> None:
    assert set(warn_only_fields()) == {"apartment", "entrance", "houseNumber"}


def test_override_and_warn_only_are_disjoint() -> None:
    assert not (set(override_fields()) & set(warn_only_fields()))
