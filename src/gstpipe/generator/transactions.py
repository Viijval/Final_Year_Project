"""Bank debits for a business: invoice payments (with their ground-truth matches) plus non-invoice decoys.

Reconciliation is made realistically hard without any OCR noise: TDS deductions, instalments,
consolidated payments, late settlement and bank narrations that truncate counterparty names.
"""

from __future__ import annotations

import random
import re
from calendar import monthrange
from dataclasses import dataclass
from datetime import date, timedelta
from decimal import Decimal

from gstpipe import catalog
from gstpipe.generator.entities import Business
from gstpipe.generator.invoices import _months
from gstpipe.gst import money
from gstpipe.schema import Invoice, Match, MatchType, Transaction, TxnKind

_KEY_BY_PATH = {catalog.category_path(k): k for k in catalog.TAXONOMY}
_TDS_RATE = {k: max((s.tds_rate for s in catalog.ITEMS_BY_CATEGORY[k]), default=Decimal("0")) for k in catalog.TAXONOMY}
_BANKS = ("HDFC", "ICIC", "SBIN", "UTIB", "KKBK", "YESB")
_PSPS = ("okhdfcbank", "okicici", "ybl", "paytm", "axl")

P_UNPAID = 0.06
P_TDS_WHEN_APPLICABLE = 0.75
P_PARTIAL = 0.07
P_CONSOLIDATE = 0.35
P_LATE = 0.12
P_INVOICE_REF_IN_NARRATION = 0.3


@dataclass
class _Pending:
    invoice: Invoice
    pay_on: date
    tds: Decimal


def bank_name(name: str, limit: int) -> str:
    cleaned = re.sub(r"[^A-Z0-9 ]", "", name.upper().replace("PVT LTD", "PVT").replace("&", "AND"))
    return re.sub(r"\s+", " ", cleaned).strip()[:limit].strip()


def _mode(rng: random.Random, amount: Decimal) -> str:
    if amount >= 200_000:
        return rng.choice(("RTGS", "RTGS", "NEFT"))
    if amount <= 50_000:
        return rng.choice(("UPI", "IMPS", "NEFT", "NEFT"))
    return rng.choice(("NEFT", "NEFT", "IMPS", "CHQ"))


def narration(rng: random.Random, mode: str, counterparty: str, remark: str = "") -> str:
    bank = rng.choice(_BANKS)
    if mode == "NEFT":
        text = f"NEFT/{bank}N{rng.randrange(10**12):012d}/{bank_name(counterparty, 22)}/{rng.choice(_BANKS)}"
    elif mode == "RTGS":
        text = f"RTGS/{bank}R{rng.randrange(10**14):014d}/{bank_name(counterparty, 24)}"
    elif mode == "IMPS":
        text = f"IMPS/P2A/{rng.randrange(10**12):012d}/{bank_name(counterparty, 18)}"
    elif mode == "UPI":
        handle = bank_name(counterparty, 14).replace(" ", "").lower()
        text = f"UPI/{rng.randrange(10**12):012d}/{handle}@{rng.choice(_PSPS)}"
    elif mode == "CHQ":
        text = f"CHQ PAID {rng.randint(100000, 999999)} {bank_name(counterparty, 20)}"
    else:
        text = f"ACH/{bank_name(counterparty, 24)}"
    return f"{text}/{remark}" if remark else text


def _plan(rng: random.Random, inv: Invoice) -> _Pending | None:
    if rng.random() < P_UNPAID:
        return None
    delay = rng.randint(30, 90) if rng.random() < P_LATE else rng.randint(-3, 10)
    pay_on = max(inv.invoice_date, inv.due_date + timedelta(days=delay))
    tds = Decimal("0")
    key = _KEY_BY_PATH[inv.category]
    if _TDS_RATE[key] and rng.random() < P_TDS_WHEN_APPLICABLE:
        tds = money(inv.taxable_total * _TDS_RATE[key] / 100)
    return _Pending(inv, pay_on, tds)


def make_transactions(rng: random.Random, business: Business, invoices: list[Invoice],
                      start: date, end: date) -> tuple[list[Transaction], list[Match]]:
    drafts: list[tuple[Transaction, list[Match]]] = []

    def add(on: date, amount: Decimal, kind: TxnKind, text: str, counterparty: str | None,
            mode: str, matches: list[Match] | None = None) -> None:
        txn = Transaction(txn_id="", business_id=business.party.party_id, split=business.split, txn_date=on,
                          amount=money(amount), mode=mode, narration=text, kind=kind, counterparty_id=counterparty)
        drafts.append((txn, matches or []))

    pending = [p for inv in invoices if (p := _plan(rng, inv)) and p.pay_on <= end]

    # Consolidate same-vendor, same-month payments without TDS into a single transfer.
    groups: dict[tuple[str, int, int], list[_Pending]] = {}
    singles: list[_Pending] = []
    for p in pending:
        if p.tds == 0 and rng.random() < P_CONSOLIDATE:
            groups.setdefault((p.invoice.supplier.party_id, p.pay_on.year, p.pay_on.month), []).append(p)
        else:
            singles.append(p)
    for batch in groups.values():
        if len(batch) == 1:
            singles.append(batch[0])
            continue
        on = max(p.pay_on for p in batch)
        total = sum((p.invoice.invoice_total for p in batch), Decimal("0"))
        supplier = batch[0].invoice.supplier
        mode = _mode(rng, total)
        matches = [Match(txn_id="", invoice_id=p.invoice.invoice_id, allocated_amount=p.invoice.invoice_total,
                         tds_amount=Decimal("0"), match_type=MatchType.consolidated) for p in batch]
        add(on, total, TxnKind.invoice_payment, narration(rng, mode, supplier.name), supplier.party_id, mode, matches)

    for p in singles:
        inv, supplier = p.invoice, p.invoice.supplier
        net = inv.invoice_total - p.tds
        remark = inv.invoice_number if rng.random() < P_INVOICE_REF_IN_NARRATION else ""
        if p.tds == 0 and net >= 20_000 and rng.random() < P_PARTIAL:
            first = money(net * Decimal(str(rng.uniform(0.3, 0.7)))).to_integral_value()
            second_on = min(end, p.pay_on + timedelta(days=rng.randint(10, 40)))
            for on, amount in ((p.pay_on, first), (second_on, net - first)):
                mode = _mode(rng, amount)
                match = Match(txn_id="", invoice_id=inv.invoice_id, allocated_amount=money(amount),
                              tds_amount=Decimal("0"), match_type=MatchType.partial)
                add(on, amount, TxnKind.invoice_payment, narration(rng, mode, supplier.name, remark),
                    supplier.party_id, mode, [match])
            continue
        mode = _mode(rng, net)
        match_type = MatchType.net_of_tds if p.tds else MatchType.exact
        match = Match(txn_id="", invoice_id=inv.invoice_id, allocated_amount=money(net),
                      tds_amount=p.tds, match_type=match_type)
        add(p.pay_on, net, TxnKind.invoice_payment, narration(rng, mode, supplier.name, remark),
            supplier.party_id, mode, [match])

    _add_decoys(rng, business, start, end, add)

    drafts.sort(key=lambda d: (d[0].txn_date, d[0].narration))
    transactions, all_matches = [], []
    for n, (txn, matches) in enumerate(drafts, start=1):
        txn_id = f"{business.party.party_id}-T{n:05d}"
        transactions.append(txn.model_copy(update={"txn_id": txn_id}))
        all_matches.extend(m.model_copy(update={"txn_id": txn_id}) for m in matches)
    return transactions, all_matches


def _add_decoys(rng: random.Random, business: Business, start: date, end: date, add) -> None:
    headcount = rng.randint(4, 40)
    payroll = money(headcount * rng.uniform(22_000, 48_000))
    emi = money(rng.choice((18_500, 42_300, 67_900, 125_000))) if rng.random() < 0.6 else None
    for year, month in _months(start, end):
        last_day = monthrange(year, month)[1]
        def on(day: int, year: int = year, month: int = month, last_day: int = last_day) -> date | None:
            d = date(year, month, min(day, last_day))
            return d if start <= d <= end else None
        if d := on(1):
            add(d, payroll * Decimal(str(rng.uniform(0.97, 1.03))), TxnKind.salary,
                f"ACH/SALARY BATCH {d:%b%y}".upper(), None, "ACH")
        if d := on(20):
            add(d, rng.uniform(8_000, 160_000), TxnKind.gst_remittance,
                f"GST/CPIN/{rng.randrange(10**14):014d}", None, "NEFT")
        if emi and (d := on(5)):
            add(d, emi, TxnKind.loan_emi, f"ACH/EMI/{rng.choice(_BANKS)} FINANCE LTD", None, "ACH")
        if d := on(rng.randint(25, 31)):
            add(d, rng.choice((17.70, 59.00, 118.00, 590.00)), TxnKind.bank_charge,
                rng.choice(("CHRG/SMS ALERT CHG", "CHRG/NEFT CHARGES", "CHRG/AMC DEBIT CARD")), None, "ACH")
