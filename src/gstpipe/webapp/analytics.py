"""Business analytics over what the pipeline has read.

Everything here is derived from invoices and bank payments that came out of the pipeline, so the
numbers a business owner sees trace back to documents rather than to hand-keyed entries.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, timedelta
from decimal import Decimal

from sqlalchemy import select
from sqlalchemy.orm import Session

from gstpipe.webapp import db
from gstpipe.webapp.verification import VERIFIED, assess

ZERO = Decimal("0")
AGEING_BUCKETS = ((0, 30, "0–30 days"), (31, 60, "31–60 days"), (61, 90, "61–90 days"),
                  (91, 10_000, "over 90 days"))


@dataclass
class Series:
    """A month-by-month line, ready to draw."""

    labels: list[str] = field(default_factory=list)
    values: list[Decimal] = field(default_factory=list)

    @property
    def peak(self) -> Decimal:
        return max(self.values) if self.values else ZERO

    @property
    def total(self) -> Decimal:
        return sum(self.values, ZERO)

    @property
    def average(self) -> Decimal:
        return (self.total / len(self.values)) if self.values else ZERO

    def points(self) -> list[tuple[str, Decimal, float]]:
        peak = self.peak or Decimal("1")
        return [(label, value, float(value / peak * 100))
                for label, value in zip(self.labels, self.values)]


def _months(start: date, end: date) -> list[tuple[int, int]]:
    out, year, month = [], start.year, start.month
    while (year, month) <= (end.year, end.month):
        out.append((year, month))
        year, month = (year + 1, 1) if month == 12 else (year, month + 1)
    return out


def _label(year: int, month: int) -> str:
    return date(year, month, 1).strftime("%b %y")


def _invoices(session: Session, business_id: str | None) -> list[db.Invoice]:
    from sqlalchemy.orm import selectinload

    statement = select(db.Invoice).options(selectinload(db.Invoice.fields),
                                           selectinload(db.Invoice.flags))
    if business_id:
        statement = statement.where(db.Invoice.business_id == business_id)
    return list(session.scalars(statement).all())


def _payments(session: Session, business_id: str | None) -> list[db.Payment]:
    statement = select(db.Payment)
    if business_id:
        statement = statement.where(db.Payment.business_id == business_id)
    return list(session.scalars(statement).all())


def build(session: Session, business_id: str | None = None, today: date | None = None) -> dict:
    invoices = _invoices(session, business_id)
    payments = _payments(session, business_id)
    usable = [i for i in invoices if i.invoice_date and i.invoice_total is not None]

    # A misread year puts an invoice decades out and would drag every chart with it. Keep to a
    # plausible window around the middle of the data and report what was set aside.
    implausible = 0
    if usable:
        ordered = sorted(i.invoice_date for i in usable)
        middle = ordered[len(ordered) // 2]
        low, high = middle - timedelta(days=730), middle + timedelta(days=200)
        plausible = [i for i in usable if low <= i.invoice_date <= high]
        implausible = len(usable) - len(plausible)
        usable = plausible

    dated = usable
    today = today or max((i.invoice_date for i in dated), default=date.today())

    spend_by_month: dict[tuple[int, int], Decimal] = defaultdict(lambda: ZERO)
    tax_by_month: dict[tuple[int, int], Decimal] = defaultdict(lambda: ZERO)
    for invoice in dated:
        key = (invoice.invoice_date.year, invoice.invoice_date.month)
        spend_by_month[key] += invoice.invoice_total or ZERO
        tax_by_month[key] += ((invoice.cgst_total or ZERO) + (invoice.sgst_total or ZERO)
                              + (invoice.igst_total or ZERO))

    cash_by_month: dict[tuple[int, int], Decimal] = defaultdict(lambda: ZERO)
    for payment in payments:
        if payment.txn_date:
            cash_by_month[(payment.txn_date.year, payment.txn_date.month)] += payment.amount or ZERO

    span = _months(min((i.invoice_date for i in dated), default=today), today)[-12:]
    spend = Series([_label(*m) for m in span], [spend_by_month[m] for m in span])
    tax = Series([_label(*m) for m in span], [tax_by_month[m] for m in span])
    cash = Series([_label(*m) for m in span], [cash_by_month[m] for m in span])

    settled = {p.matched_invoice_id for p in payments if p.matched_invoice_id}
    outstanding = [i for i in dated if i.id not in settled]
    ageing = []
    for low, high, label in AGEING_BUCKETS:
        bucket = [i for i in outstanding
                  if low <= (today - (i.due_date or i.invoice_date)).days <= high]
        ageing.append({"label": label, "count": len(bucket),
                       "value": sum((i.invoice_total or ZERO for i in bucket), ZERO)})

    by_supplier: dict[str, list[Decimal]] = defaultdict(list)
    by_category: dict[str, Decimal] = defaultdict(lambda: ZERO)
    for invoice in dated:
        by_supplier[invoice.supplier_name or "Unknown supplier"].append(invoice.invoice_total or ZERO)
        by_category[(invoice.category or "Uncategorised").split(" > ")[-1]] += invoice.invoice_total or ZERO

    total_spend = sum((i.invoice_total or ZERO for i in dated), ZERO) or Decimal("1")
    suppliers = sorted(
        ({"name": name, "invoices": len(values), "value": sum(values, ZERO),
          "share": float(sum(values, ZERO) / total_spend * 100)}
         for name, values in by_supplier.items()),
        key=lambda row: row["value"], reverse=True)

    # How long the business actually takes to pay, measured against the invoice date.
    invoice_dates = {i.id: i.invoice_date for i in dated}
    lags = [(p.txn_date - invoice_dates[p.matched_invoice_id]).days
            for p in payments
            if p.matched_invoice_id in invoice_dates and p.txn_date]
    lags = [lag for lag in lags if lag >= 0]

    # Every figure on the page should be able to say how much of itself is trustworthy.
    verified_value = ZERO
    verified_count = 0
    for invoice in dated:
        judgement = assess(invoice, signed_off=invoice.status == "approved",
                           traced={f.field for f in invoice.fields if f.region})
        if judgement.state == VERIFIED:
            verified_value += invoice.invoice_total or ZERO
            verified_count += 1

    return {
        "as_of": today,
        "verified_value": verified_value,
        "verified_count": verified_count,
        "verified_share": float(verified_count / len(dated) * 100) if dated else 0.0,
        "spend": spend,
        "tax": tax,
        "cash": cash,
        "invoice_count": len(dated),
        "total_spend": sum((i.invoice_total or ZERO for i in dated), ZERO),
        "input_credit": tax.total,
        "outstanding_count": len(outstanding),
        "outstanding_value": sum((i.invoice_total or ZERO for i in outstanding), ZERO),
        "overdue_value": sum((row["value"] for row in ageing[1:]), ZERO),
        "ageing": ageing,
        "suppliers": suppliers[:10],
        "supplier_count": len(by_supplier),
        "categories": sorted(by_category.items(), key=lambda kv: kv[1], reverse=True)[:10],
        "average_days_to_pay": round(sum(lags) / len(lags)) if lags else None,
        "largest_invoice": max(dated, key=lambda i: i.invoice_total or ZERO, default=None),
        "needs_review": sum(1 for i in invoices if i.status == "needs_review"),
        "coverage": {
            "invoices": len(invoices),
            "usable": len(dated),
            "no_date_or_total": sum(1 for i in invoices
                                    if not i.invoice_date or i.invoice_total is None),
            "implausible_date": implausible,
            "no_tax": sum(1 for i in dated if i.cgst_total is None and i.igst_total is None),
        },
    }
