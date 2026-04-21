"""Basic tests for the anchor-label digit parser."""

from __future__ import annotations

from form_extraction.core.digits import parse_date, parse_id


def test_parse_date_reads_clean_8_digit_box() -> None:
    md = "תאריך הפגיעה\n\n[1]6 0| 4 2 | 0 | 2 | 2|\nשנה חודש יום"
    assert parse_date(md, "dateOfInjury") == ("16", "04", "2022")


def test_parse_id_reads_9_digit_block() -> None:
    md = "ת.ז.\n0 2 2 4 5 6 1 2 0"
    assert parse_id(md) == "022456120"
