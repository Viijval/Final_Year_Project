"""Scoring an extraction against ground truth, field by field.

A field is correct, wrong or missing. Those are different failures: a missing amount stops the
pipeline, a wrong one carries a plausible lie downstream, which is what the propagation study is about.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal
from enum import Enum

from gstpipe.extract.records import FIELDS, MONEY_FIELDS, TEXT_FIELDS, ExtractedInvoice
from gstpipe.schema import Invoice

MONEY_TOLERANCE = Decimal("0.01")


class Outcome(str, Enum):
    correct = "correct"
    wrong = "wrong"
    missing = "missing"


def truth_values(invoice: Invoice) -> dict[str, object]:
    return {
        "supplier_name": invoice.supplier.name, "supplier_gstin": invoice.supplier.gstin,
        "recipient_name": invoice.recipient.name, "recipient_gstin": invoice.recipient.gstin,
        "invoice_number": invoice.invoice_number, "invoice_date": invoice.invoice_date,
        "due_date": invoice.due_date, "place_of_supply": invoice.place_of_supply,
        "taxable_total": invoice.taxable_total, "cgst_total": invoice.cgst_total,
        "sgst_total": invoice.sgst_total, "igst_total": invoice.igst_total,
        "invoice_total": invoice.invoice_total,
    }


def normalise(field: str, value):
    if value is None:
        return None
    if field in TEXT_FIELDS:
        text = re.sub(r"[^a-z0-9 ]", "", str(value).casefold())
        return re.sub(r"\s+", " ", text).strip()
    if field in MONEY_FIELDS:
        return Decimal(str(value))
    return str(value).strip().upper().replace(" ", "")


def compare_field(field: str, truth, predicted) -> Outcome:
    if predicted is None:
        # A tax component that does not apply is simply not printed: an intra-state invoice shows no
        # IGST line. Absence on the document means zero, so a zero truth is correctly read as absent.
        if field in MONEY_FIELDS and truth is not None and Decimal(str(truth)) == 0:
            return Outcome.correct
        return Outcome.missing
    a, b = normalise(field, truth), normalise(field, predicted)
    if field in MONEY_FIELDS:
        return Outcome.correct if abs(a - b) <= MONEY_TOLERANCE else Outcome.wrong
    return Outcome.correct if a == b else Outcome.wrong


def compare(invoice: Invoice, extracted: ExtractedInvoice) -> dict[str, Outcome]:
    truth = truth_values(invoice)
    return {f: compare_field(f, truth[f], getattr(extracted, f)) for f in FIELDS}


@dataclass
class FieldScore:
    correct: int = 0
    wrong: int = 0
    missing: int = 0

    @property
    def total(self) -> int:
        return self.correct + self.wrong + self.missing

    @property
    def accuracy(self) -> float:
        """Share of fields recovered correctly, counting a miss as a failure."""
        return self.correct / self.total if self.total else 0.0

    @property
    def precision(self) -> float:
        """Of the fields it committed to, how many were right."""
        attempted = self.correct + self.wrong
        return self.correct / attempted if attempted else 0.0

    @property
    def recall(self) -> float:
        return self.accuracy

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if p + r else 0.0

    def add(self, outcome: Outcome) -> None:
        setattr(self, outcome.value, getattr(self, outcome.value) + 1)

    def as_row(self) -> dict[str, float | int]:
        return {"correct": self.correct, "wrong": self.wrong, "missing": self.missing,
                "accuracy": round(self.accuracy, 4), "precision": round(self.precision, 4),
                "f1": round(self.f1, 4)}


def score(results: list[dict[str, Outcome]]) -> tuple[FieldScore, dict[str, FieldScore]]:
    overall, per_field = FieldScore(), {f: FieldScore() for f in FIELDS}
    for row in results:
        for field, outcome in row.items():
            overall.add(outcome)
            per_field[field].add(outcome)
    return overall, per_field
