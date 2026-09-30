"""Planted anomalies with labels. Every anomalous invoice is a new record, so existing matches stay valid.

About half of the anomalous invoices are paid like any other, so "unpaid" cannot become a shortcut
feature for spotting anomalies.
"""

from __future__ import annotations

import random
import re
from datetime import date, timedelta
from decimal import Decimal
from statistics import median

from gstpipe.generator.entities import Business
from gstpipe.generator.invoices import finalise
from gstpipe.generator.transactions import _mode, narration
from gstpipe.gst import compute_tax, money
from gstpipe.schema import (Anomaly, AnomalyType, Invoice, LineItem, Match, MatchType, Transaction,
                            TxnKind)

INVOICE_ANOMALY_WEIGHTS: dict[AnomalyType, int] = {
    AnomalyType.duplicate_invoice: 3,
    AnomalyType.near_duplicate: 2,
    AnomalyType.amount_outlier: 3,
    AnomalyType.gst_arithmetic_error: 2,
    AnomalyType.wrong_tax_type: 1,
    AnomalyType.invalid_gstin: 1,
    AnomalyType.due_before_invoice_date: 1,
}
OUTLIER_MULTIPLE = (6.0, 12.0)  # outlier taxable value relative to the supplier's median invoice
P_ANOMALY_PAID = 0.5


def _retax(line: LineItem, intra: bool, unit_price: Decimal | None = None) -> LineItem:
    price = line.unit_price if unit_price is None else money(unit_price)
    taxable = money(price * line.quantity)
    tax = compute_tax(taxable, line.gst_rate, intra)
    return line.model_copy(update={"unit_price": price, "taxable_value": taxable,
                                   "cgst": tax.cgst, "sgst": tax.sgst, "igst": tax.igst})


def _fresh_number(rng: random.Random, number: str, taken: set[str]) -> str:
    match = re.search(r"(\d+)$", number)
    for _ in range(100):
        if match:
            width = len(match.group(1))
            candidate = number[:match.start(1)] + f"{int(match.group(1)) + rng.randint(3, 900):0{width}d}"
        else:
            candidate = f"{number}-{rng.randint(10, 99)}"
        if candidate not in taken:
            return candidate
    raise RuntimeError(f"could not derive a fresh invoice number from {number!r}")


def _mutate(rng: random.Random, kind: AnomalyType, src: Invoice, new_id: str, taken: set[str],
            end: date, supplier_median: Decimal) -> tuple[Invoice, str]:
    intra = src.supplier.state_code == src.place_of_supply
    shift = timedelta(days=rng.randint(0, 5))
    base = src.model_copy(update={"invoice_id": new_id,
                                  "invoice_number": _fresh_number(rng, src.invoice_number, taken),
                                  "invoice_date": min(end, src.invoice_date + shift),
                                  "due_date": src.due_date + shift})

    if kind is AnomalyType.duplicate_invoice:
        dup = src.model_copy(update={"invoice_id": new_id,
                                     "invoice_date": min(end, src.invoice_date + timedelta(days=rng.randint(1, 20)))})
        return dup, f"re-submission of {src.invoice_id} with identical number {src.invoice_number}"

    if kind is AnomalyType.near_duplicate:
        return base, f"same supplier and total as {src.invoice_id}, different number, within {shift.days} days"

    if kind is AnomalyType.amount_outlier:
        multiple = Decimal(str(round(rng.uniform(*OUTLIER_MULTIPLE), 2)))
        factor = supplier_median * multiple / src.taxable_total
        lines = [_retax(li, intra, li.unit_price * factor) for li in src.line_items]
        return (finalise(base.model_copy(update={"line_items": lines})),
                f"taxable value about {multiple}x the supplier median")

    if kind is AnomalyType.wrong_tax_type:
        lines = [_retax(li, not intra) for li in src.line_items]
        detail = "IGST charged on intra-state supply" if intra else "CGST+SGST charged on inter-state supply"
        return finalise(base.model_copy(update={"line_items": lines})), detail

    if kind is AnomalyType.invalid_gstin:
        g = src.supplier.gstin
        bad_check = next(c for c in "ABCDEFGHJKLMNPQRSTUVWXYZ0123456789" if c != g[-1])
        supplier = src.supplier.model_copy(update={"gstin": g[:-1] + bad_check})
        return base.model_copy(update={"supplier": supplier}), f"supplier GSTIN checksum invalid ({g[:-1]}{bad_check})"

    if kind is AnomalyType.due_before_invoice_date:
        return (base.model_copy(update={"due_date": base.invoice_date - timedelta(days=rng.randint(3, 30))}),
                "due date precedes invoice date")

    # gst_arithmetic_error: a document typo that breaks internal consistency
    if rng.random() < 0.5:
        delta = money(rng.choice((9, 90, 900, -90, -900)))
        return (base.model_copy(update={"invoice_total": base.invoice_total + delta}),
                f"stated total differs from taxable + tax + round-off by {delta}")
    component = "igst_total" if base.igst_total else "cgst_total"
    wrong = money(getattr(base, component) * 2)
    return (base.model_copy(update={component: wrong, "invoice_total": base.invoice_total + wrong / 2}),
            f"{component} doubled relative to taxable value and rate")


def inject_anomalies(rng: random.Random, business: Business, invoices: list[Invoice],
                     transactions: list[Transaction], matches: list[Match], end: date, rate: float
                     ) -> tuple[list[Invoice], list[Transaction], list[Match], list[Anomaly]]:
    bid = business.party.party_id
    invoices, transactions, matches = list(invoices), list(transactions), list(matches)
    anomalies: list[Anomaly] = []
    numbers: dict[str, set[str]] = {}
    history: dict[str, list[Decimal]] = {}
    for inv in invoices:
        numbers.setdefault(inv.supplier.party_id, set()).add(inv.invoice_number)
        history.setdefault(inv.supplier.party_id, []).append(inv.taxable_total)
    supplier_median = {party_id: median(values) for party_id, values in history.items()}

    def label(entity_type: str, entity_id: str, kind: AnomalyType, detail: str) -> None:
        anomalies.append(Anomaly(anomaly_id=f"{bid}-A{len(anomalies) + 1:04d}", entity_type=entity_type,
                                 entity_id=entity_id, anomaly_type=kind, detail=detail))

    def next_txn_id() -> str:
        return f"{bid}-T{len(transactions) + 1:05d}"

    kinds, weights = zip(*INVOICE_ANOMALY_WEIGHTS.items())
    sources = rng.sample(invoices, k=min(len(invoices), round(len(invoices) * rate)))
    for src in sources:
        kind = rng.choices(kinds, weights)[0]
        new_id = f"{bid}-I{len(invoices) + 1:05d}"
        taken = numbers.setdefault(src.supplier.party_id, set())
        anomalous, detail = _mutate(rng, kind, src, new_id, taken, end, supplier_median[src.supplier.party_id])
        taken.add(anomalous.invoice_number)
        invoices.append(anomalous)
        label("invoice", new_id, kind, detail)

        pay_on = max(anomalous.invoice_date, anomalous.due_date) + timedelta(days=rng.randint(0, 10))
        if rng.random() < P_ANOMALY_PAID and pay_on <= end:
            mode = _mode(rng, anomalous.invoice_total)
            txn_id = next_txn_id()
            transactions.append(Transaction(
                txn_id=txn_id, business_id=bid, split=business.split, txn_date=pay_on,
                amount=anomalous.invoice_total, mode=mode, narration=narration(rng, mode, anomalous.supplier.name),
                kind=TxnKind.invoice_payment, counterparty_id=anomalous.supplier.party_id))
            matches.append(Match(txn_id=txn_id, invoice_id=new_id, allocated_amount=anomalous.invoice_total,
                                 tds_amount=Decimal("0"), match_type=MatchType.exact))

    paid = [t for t in transactions if t.kind is TxnKind.invoice_payment and t.counterparty_id]
    for template in rng.sample(paid, k=min(len(paid), max(1, round(len(paid) * rate / 2)))):
        vendor = next(v for v in business.vendors if v.party.party_id == template.counterparty_id)
        on = min(end, template.txn_date + timedelta(days=rng.randint(3, 25)))
        amount = money(template.amount * Decimal(str(round(rng.uniform(0.6, 1.4), 2))))
        mode = _mode(rng, amount)
        txn_id = next_txn_id()
        transactions.append(Transaction(
            txn_id=txn_id, business_id=bid, split=business.split, txn_date=on, amount=amount, mode=mode,
            narration=narration(rng, mode, vendor.party.name), kind=TxnKind.unbilled_vendor_payment,
            counterparty_id=vendor.party.party_id))
        label("transaction", txn_id, AnomalyType.unbilled_payment, f"payment to {vendor.party.name} with no invoice")

    invoices.sort(key=lambda i: (i.invoice_date, i.invoice_id))
    transactions.sort(key=lambda t: (t.txn_date, t.txn_id))
    return invoices, transactions, matches, anomalies
