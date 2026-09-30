import random
from datetime import date
from decimal import Decimal

import pytest

from gstpipe.extract.evaluate import Outcome, compare
from gstpipe.generator.entities import make_business
from gstpipe.generator.invoices import make_invoices
from gstpipe.inject.corrupt import InjectionReport, corrupt, from_invoice

START, END = date(2025, 4, 1), date(2026, 3, 31)


@pytest.fixture(scope="module")
def invoices():
    rng = random.Random(77)
    return make_invoices(rng, make_business(rng, "B000", "train", set()), START, END)


def test_a_perfect_extraction_scores_perfectly(invoices):
    for invoice in invoices[:25]:
        outcomes = compare(invoice, from_invoice(invoice))
        assert all(o is Outcome.correct for o in outcomes.values()), invoice.invoice_id


def test_zero_rate_leaves_everything_untouched(invoices):
    clean = from_invoice(invoices[0])
    assert corrupt(clean, 0.0, random.Random(1)) == clean


def test_realised_rate_tracks_the_requested_rate(invoices):
    report = InjectionReport(0.3)
    rng = random.Random(4)
    for invoice in invoices[:120]:
        corrupt(from_invoice(invoice), 0.3, rng, report)
    assert 0.24 <= report.realised_rate <= 0.36
    assert report.fields_corrupted > 0


def test_more_corruption_means_lower_field_accuracy(invoices):
    def accuracy(rate: float) -> float:
        rng = random.Random(9)
        correct = total = 0
        for invoice in invoices[:80]:
            outcomes = compare(invoice, corrupt(from_invoice(invoice), rate, rng))
            correct += sum(o is Outcome.correct for o in outcomes.values())
            total += len(outcomes)
        return correct / total

    scores = [accuracy(r) for r in (0.0, 0.1, 0.3)]
    assert scores == sorted(scores, reverse=True)
    assert scores[0] == 1.0


def test_injection_is_reproducible(invoices):
    clean = from_invoice(invoices[3])
    a = corrupt(clean, 0.4, random.Random(11))
    b = corrupt(clean, 0.4, random.Random(11))
    assert a == b


def test_corrupted_values_keep_their_types(invoices):
    rng = random.Random(6)
    for invoice in invoices[:40]:
        damaged = corrupt(from_invoice(invoice), 0.5, rng)
        assert damaged.invoice_total is None or isinstance(damaged.invoice_total, Decimal)
        assert damaged.invoice_date is None or isinstance(damaged.invoice_date, date)
        assert damaged.supplier_name is None or isinstance(damaged.supplier_name, str)


def test_line_items_can_be_dropped_but_never_invented(invoices):
    rng = random.Random(2)
    for invoice in invoices[:60]:
        damaged = corrupt(from_invoice(invoice), 0.5, rng)
        assert len(damaged.line_items) <= len(invoice.line_items)
