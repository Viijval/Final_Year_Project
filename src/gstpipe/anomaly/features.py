"""Per-invoice signals for anomaly review.

Two families: internal consistency checks the document must satisfy on its own (tax arithmetic,
a valid GSTIN, a due date after the invoice date), and behavioural comparisons against the
supplier's own history (an amount far outside its usual range, a repeated number).

Under extraction noise both families degrade, but differently: a consistency check can fail because
the document is wrong *or* because a digit was misread, and the stage cannot tell the two apart.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal
from statistics import median

from gstpipe.gst import SLABS_AFTER, SLABS_BEFORE, STATES, is_valid_gstin

FEATURE_NAMES = ("log_amount", "amount_over_vendor_median", "amount_over_vendor_max",
                 "duplicate_number", "near_duplicate", "gst_inconsistent", "tax_type_inconsistent",
                 "tax_split_uneven", "gstin_invalid", "due_before_invoice", "days_since_vendor_last",
                 "n_lines", "missing_fields")

CONSISTENCY_TOLERANCE = Decimal("0.5")
VALID_SLABS = sorted(set(SLABS_BEFORE) | set(SLABS_AFTER))


@dataclass
class InvoiceSignals:
    invoice_id: str
    business_id: str
    supplier_key: str
    invoice_date: date | None
    amount: Decimal | None
    features: dict[str, float] = field(default_factory=dict)

    @property
    def vector(self) -> list[float]:
        return [self.features[name] for name in FEATURE_NAMES]

    @property
    def structural_failures(self) -> list[str]:
        return [name for name in ("duplicate_number", "gst_inconsistent", "tax_type_inconsistent",
                                  "tax_split_uneven", "implied_rate_invalid",
                                  "gstin_invalid", "due_before_invoice")
                if self.features.get(name, 0.0) >= 1.0]


@dataclass(frozen=True)
class InvoiceRecord:
    """The fields anomaly detection needs, from ground truth or from Stage 1 output."""

    invoice_id: str
    business_id: str
    supplier_name: str | None
    supplier_gstin: str | None
    invoice_number: str | None
    invoice_date: date | None
    due_date: date | None
    place_of_supply: str | None
    taxable_total: Decimal | None
    cgst_total: Decimal | None
    sgst_total: Decimal | None
    igst_total: Decimal | None
    invoice_total: Decimal | None
    round_off: Decimal | None
    n_lines: int


def _supplier_key(record: InvoiceRecord) -> str:
    """Group a supplier's history by GSTIN when readable, else by name."""
    if record.supplier_gstin and len(record.supplier_gstin) >= 12:
        return record.supplier_gstin[:12].upper()
    return (record.supplier_name or "unknown").casefold().strip()


def _gst_inconsistent(record: InvoiceRecord) -> float:
    parts = (record.taxable_total, record.cgst_total, record.sgst_total, record.igst_total,
             record.invoice_total)
    if any(p is None for p in parts):
        return 0.0  # cannot judge a document whose numbers were not recovered
    stated = record.taxable_total + record.cgst_total + record.sgst_total + record.igst_total
    stated += record.round_off or Decimal("0")
    return float(abs(stated - record.invoice_total) > CONSISTENCY_TOLERANCE)


def _tax_type_inconsistent(record: InvoiceRecord) -> float:
    supplier_state = (record.supplier_gstin or "")[:2]
    if supplier_state not in STATES or record.place_of_supply not in STATES:
        return 0.0
    if None in (record.cgst_total, record.sgst_total, record.igst_total):
        return 0.0
    intra = supplier_state == record.place_of_supply
    if intra:
        return float(record.igst_total > 0)
    return float(record.cgst_total + record.sgst_total > 0)


def _tax_split_uneven(record: InvoiceRecord) -> float:
    """On an intra-state supply CGST and SGST are half the rate each, so they must be equal.

    A sum check alone cannot catch a component computed at the wrong rate: if the stated total is
    raised to match, the invoice still adds up while the split is plainly wrong.
    """
    if record.cgst_total is None or record.sgst_total is None:
        return 0.0
    supplier_state = (record.supplier_gstin or "")[:2]
    if supplier_state not in STATES or record.place_of_supply not in STATES:
        return 0.0
    if supplier_state != record.place_of_supply:
        return 0.0
    if record.cgst_total == 0 and record.sgst_total == 0:
        return 0.0
    return float(abs(record.cgst_total - record.sgst_total) > CONSISTENCY_TOLERANCE)


def _implied_rate_invalid(record: InvoiceRecord) -> float:
    """Total tax over taxable value should land on a GST slab.

    This catches a component computed at the wrong rate even when the stated total was adjusted to
    match it. Assumes one rate per invoice, which holds for this dataset; an invoice genuinely
    mixing slabs would blend to a rate between them and could trip this check.
    """
    parts = (record.taxable_total, record.cgst_total, record.sgst_total, record.igst_total)
    if any(p is None for p in parts) or record.taxable_total <= 0:
        return 0.0
    tax = record.cgst_total + record.sgst_total + record.igst_total
    implied = float(tax / record.taxable_total * 100)
    return float(all(abs(implied - float(slab)) > 0.6 for slab in VALID_SLABS))


def build(records: list[InvoiceRecord]) -> list[InvoiceSignals]:
    """Signals are relative to each supplier's own history, so they must be built as a batch."""
    history: dict[str, list[Decimal]] = {}
    dates: dict[str, list[date]] = {}
    for record in records:
        key = _supplier_key(record)
        if record.invoice_total is not None:
            history.setdefault(key, []).append(record.invoice_total)
        if record.invoice_date is not None:
            dates.setdefault(key, []).append(record.invoice_date)

    medians = {k: median(v) for k, v in history.items() if v}
    maxima = {k: max(v) for k, v in history.items() if v}

    seen_numbers: set[tuple[str, str]] = set()
    seen_amount_date: dict[tuple[str, str], list[date]] = {}
    # Repetition checks need chronological order, but callers align scores with their own input
    # order, so results are collected by id and returned in the order they arrived.
    by_id: dict[str, InvoiceSignals] = {}
    for record in sorted(records, key=lambda r: (r.invoice_date or date.min, r.invoice_id)):
        key = _supplier_key(record)
        amount = record.invoice_total
        vendor_median = medians.get(key)
        vendor_max = maxima.get(key)

        number_key = (key, (record.invoice_number or "").upper())
        duplicate = float(record.invoice_number is not None and number_key in seen_numbers)
        seen_numbers.add(number_key)

        amount_key = (key, f"{amount:.0f}" if amount is not None else "")
        near_duplicate = 0.0
        if amount is not None and record.invoice_date is not None:
            previous = seen_amount_date.setdefault(amount_key, [])
            near_duplicate = float(any(abs((record.invoice_date - p).days) <= 7 for p in previous))
            previous.append(record.invoice_date)

        prior_dates = [d for d in dates.get(key, []) if record.invoice_date and d < record.invoice_date]
        gap = (record.invoice_date - max(prior_dates)).days if prior_dates and record.invoice_date else 0

        missing = sum(getattr(record, f) is None for f in
                      ("supplier_gstin", "invoice_number", "invoice_date", "invoice_total",
                       "taxable_total"))

        features = {
            "log_amount": float(amount) ** 0.25 if amount and amount > 0 else 0.0,
            "amount_over_vendor_median": float(amount / vendor_median) if amount and vendor_median else 1.0,
            "amount_over_vendor_max": float(amount / vendor_max) if amount and vendor_max else 1.0,
            "duplicate_number": duplicate,
            "near_duplicate": near_duplicate,
            "gst_inconsistent": _gst_inconsistent(record),
            "tax_type_inconsistent": _tax_type_inconsistent(record),
            "tax_split_uneven": _tax_split_uneven(record),
            "implied_rate_invalid": _implied_rate_invalid(record),
            "gstin_invalid": float(bool(record.supplier_gstin) and not is_valid_gstin(record.supplier_gstin)),
            "due_before_invoice": float(record.due_date is not None and record.invoice_date is not None
                                        and record.due_date < record.invoice_date),
            "days_since_vendor_last": min(gap, 180) / 180,
            "n_lines": float(record.n_lines),
            "missing_fields": float(missing),
        }
        by_id[record.invoice_id] = InvoiceSignals(record.invoice_id, record.business_id, key,
                                                  record.invoice_date, amount, features)
    return [by_id[record.invoice_id] for record in records]
