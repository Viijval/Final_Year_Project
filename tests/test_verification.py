"""Whether an invoice can be signed off, and whether the reasons are honest."""

from dataclasses import dataclass
from datetime import date
from decimal import Decimal

import pytest

from gstpipe.webapp.verification import (NEEDS_CHECK, UNREADABLE, VERIFIED, assess, summarise)


@dataclass
class FakeFlag:
    detail: str
    resolved: bool = False


@dataclass
class FakeInvoice:
    invoice_number: str | None = "INV-1"
    invoice_date: date | None = date(2026, 1, 1)
    invoice_total: Decimal | None = Decimal("118000.00")
    taxable_total: Decimal | None = Decimal("100000.00")
    cgst_total: Decimal | None = Decimal("9000.00")
    sgst_total: Decimal | None = Decimal("9000.00")
    igst_total: Decimal | None = Decimal("0.00")
    supplier_gstin: str | None = "27AAPFU0939F1ZV"
    supplier_name: str | None = "Acme Traders"
    extraction_confidence: float = 0.95
    flags: list = None

    def __post_init__(self):
        self.flags = self.flags if self.flags is not None else []


ALL_TRACED = {"invoice_number", "invoice_date", "invoice_total", "supplier_gstin"}


def test_a_clean_invoice_verifies():
    result = assess(FakeInvoice(), traced=ALL_TRACED)
    assert result.state is VERIFIED or result.state == VERIFIED
    assert result.failed == []
    assert result.score == 1.0


@pytest.mark.parametrize("field", ["invoice_number", "invoice_date", "invoice_total"])
def test_a_missing_essential_makes_it_unreadable(field):
    result = assess(FakeInvoice(**{field: None}), traced=ALL_TRACED)
    assert result.state == UNREADABLE
    assert any(field.replace("_", " ") in c.detail for c in result.failed)


def test_no_supplier_at_all_is_unreadable():
    result = assess(FakeInvoice(supplier_gstin=None, supplier_name=None), traced=ALL_TRACED)
    assert result.state == UNREADABLE


def test_a_name_alone_is_enough_to_identify_a_supplier():
    result = assess(FakeInvoice(supplier_gstin=None), traced=ALL_TRACED)
    assert result.state != UNREADABLE


def test_arithmetic_that_does_not_add_up_needs_checking():
    result = assess(FakeInvoice(invoice_total=Decimal("125000.00")), traced=ALL_TRACED)
    assert result.state == NEEDS_CHECK
    assert any(c.name == "Adds up" and not c.ok for c in result.checks)


def test_a_poor_scan_needs_checking_even_when_everything_parsed():
    result = assess(FakeInvoice(extraction_confidence=0.4), traced=ALL_TRACED)
    assert result.state == NEEDS_CHECK
    assert any(c.name == "Read confidently" and not c.ok for c in result.checks)


def test_an_open_flag_blocks_verification_but_a_resolved_one_does_not():
    flagged = assess(FakeInvoice(flags=[FakeFlag("duplicate number")]), traced=ALL_TRACED)
    assert flagged.state == NEEDS_CHECK
    cleared = assess(FakeInvoice(flags=[FakeFlag("duplicate number", resolved=True)]),
                     traced=ALL_TRACED)
    assert cleared.state == VERIFIED


def test_values_that_cannot_be_located_on_the_page_block_verification():
    result = assess(FakeInvoice(), traced=set())
    assert result.state == NEEDS_CHECK
    assert any(c.name == "Traceable to the document" and not c.ok for c in result.checks)


def test_a_person_can_sign_off_something_the_checks_dislike():
    """A CA overrules the machine; the reasons stay visible underneath."""
    result = assess(FakeInvoice(extraction_confidence=0.2), signed_off=True, traced=ALL_TRACED)
    assert result.state == VERIFIED
    assert result.failed, "the failing checks should still be reported"


def test_signing_off_cannot_rescue_an_unreadable_document():
    result = assess(FakeInvoice(invoice_total=None), signed_off=True, traced=ALL_TRACED)
    assert result.state == UNREADABLE


def test_summary_splits_count_and_value_by_state():
    rows = [(assess(FakeInvoice(), traced=ALL_TRACED), Decimal("100")),
            (assess(FakeInvoice(extraction_confidence=0.1), traced=ALL_TRACED), Decimal("200")),
            (assess(FakeInvoice(invoice_date=None), traced=ALL_TRACED), Decimal("400"))]
    out = summarise(rows)
    assert out[VERIFIED]["count"] == 1 and out[VERIFIED]["value"] == Decimal("100")
    assert out[NEEDS_CHECK]["value"] == Decimal("200")
    assert out[UNREADABLE]["value"] == Decimal("400")
    assert sum(b["share"] for b in out.values()) == pytest.approx(100.0)
