"""Signals for one candidate (transaction, invoice) pair.

Matching leans on four partly independent signals: amount, date, the supplier name inside the bank
narration, and any invoice number quoted there. That redundancy is why reconciliation might absorb
extraction noise rather than amplify it -- the question the propagation study settles.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from decimal import Decimal

from rapidfuzz import fuzz

from gstpipe.schema import Transaction

MAX_LAG_DAYS = 150
TDS_BAND = (Decimal("0.85"), Decimal("1.0"))


@dataclass(frozen=True)
class Candidate:
    txn_id: str
    invoice_id: str
    features: dict[str, float]

    @property
    def vector(self) -> list[float]:
        return [self.features[k] for k in FEATURE_NAMES]


FEATURE_NAMES = ("amount_ratio", "amount_exact", "amount_close", "tds_plausible", "log_amount_gap",
                 "days_after_invoice", "days_after_due", "within_terms", "vendor_similarity",
                 "vendor_partial", "invoice_number_hit", "narration_digits_hit")


def _normalise(text: str) -> str:
    return re.sub(r"[^A-Z0-9 ]", " ", (text or "").upper())


def _number_hit(invoice_number: str | None, narration: str) -> float:
    if not invoice_number:
        return 0.0
    cleaned = _normalise(invoice_number).replace(" ", "")
    narration_clean = _normalise(narration).replace(" ", "")
    if cleaned and cleaned in narration_clean:
        return 1.0
    digits = re.sub(r"\D", "", invoice_number)
    return 0.5 if len(digits) >= 4 and digits in narration_clean else 0.0


def build(txn: Transaction, invoice_id: str, supplier_name: str | None, invoice_total: Decimal | None,
          invoice_date: date | None, due_date: date | None, invoice_number: str | None) -> Candidate | None:
    """None when the pair is impossible: a payment cannot precede its invoice by much."""
    if invoice_date is None or invoice_total is None or invoice_total <= 0:
        return None
    lag = (txn.txn_date - invoice_date).days
    if lag < -3 or lag > MAX_LAG_DAYS:
        return None

    ratio = float(txn.amount / invoice_total)
    gap = abs(float(txn.amount - invoice_total))
    narration = txn.narration
    similarity = fuzz.token_set_ratio(_normalise(supplier_name or ""), _normalise(narration)) / 100
    partial = fuzz.partial_ratio(_normalise(supplier_name or ""), _normalise(narration)) / 100

    features = {
        "amount_ratio": min(ratio, 3.0),
        "amount_exact": float(gap <= 0.01),
        "amount_close": float(gap <= 1.0),
        "tds_plausible": float(TDS_BAND[0] <= Decimal(str(round(ratio, 4))) <= TDS_BAND[1]),
        "log_amount_gap": min(6.0, (gap + 1) ** 0.25),
        "days_after_invoice": lag / MAX_LAG_DAYS,
        "days_after_due": ((txn.txn_date - due_date).days / MAX_LAG_DAYS) if due_date else 0.0,
        "within_terms": float(due_date is not None and txn.txn_date <= due_date),
        "vendor_similarity": similarity,
        "vendor_partial": partial,
        "invoice_number_hit": _number_hit(invoice_number, narration),
        "narration_digits_hit": float(bool(re.search(r"\d{4,}", narration))),
    }
    return Candidate(txn_id=txn.txn_id, invoice_id=invoice_id, features=features)
