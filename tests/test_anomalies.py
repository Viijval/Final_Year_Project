import random
from collections import defaultdict
from datetime import date
from decimal import Decimal
from statistics import median

import pytest

from gstpipe.generator.anomalies import INVOICE_ANOMALY_WEIGHTS, inject_anomalies
from gstpipe.generator.entities import make_business
from gstpipe.generator.invoices import make_invoices
from gstpipe.generator.transactions import make_transactions
from gstpipe.gst import is_valid_gstin
from gstpipe.schema import AnomalyType, TxnKind

START, END = date(2025, 4, 1), date(2026, 3, 31)


@pytest.fixture(scope="module")
def world():
    rng = random.Random(23)
    used: set[str] = set()
    invoices, txns, matches, anomalies = [], [], [], []
    for i in range(10):
        b = make_business(rng, f"B{i:03d}", "train", used)
        inv = make_invoices(rng, b, START, END)
        t, m = make_transactions(rng, b, inv, START, END)
        inv, t, m, a = inject_anomalies(rng, b, inv, t, m, END, rate=0.12)
        invoices += inv; txns += t; matches += m; anomalies += a
    return invoices, txns, matches, anomalies


def _structural_problem(inv) -> bool:
    intra = inv.supplier.state_code == inv.place_of_supply
    header = inv.taxable_total + inv.cgst_total + inv.sgst_total + inv.igst_total + inv.round_off
    line_tax = {k: sum(getattr(li, k) for li in inv.line_items) for k in ("cgst", "sgst", "igst")}
    return (
        header != inv.invoice_total
        or any(getattr(inv, f"{k}_total") != v for k, v in line_tax.items())
        or (intra and inv.igst_total > 0) or (not intra and inv.cgst_total + inv.sgst_total > 0)
        or not is_valid_gstin(inv.supplier.gstin)
        or inv.due_date < inv.invoice_date
    )


def test_every_label_points_at_a_real_record(world):
    invoices, txns, _, anomalies = world
    ids = {"invoice": {i.invoice_id for i in invoices}, "transaction": {t.txn_id for t in txns}}
    assert anomalies
    assert all(a.entity_id in ids[a.entity_type] for a in anomalies)
    assert len({a.anomaly_id for a in anomalies}) == len(anomalies)


def test_all_anomaly_types_are_planted(world):
    *_, anomalies = world
    assert {a.anomaly_type for a in anomalies} == set(INVOICE_ANOMALY_WEIGHTS) | {AnomalyType.unbilled_payment}


def test_structural_anomalies_are_actually_broken(world):
    invoices, _, _, anomalies = world
    by_id = {i.invoice_id: i for i in invoices}
    structural = {AnomalyType.gst_arithmetic_error, AnomalyType.wrong_tax_type,
                  AnomalyType.invalid_gstin, AnomalyType.due_before_invoice_date}
    for a in anomalies:
        if a.anomaly_type in structural:
            assert _structural_problem(by_id[a.entity_id]), a


def test_no_unlabelled_invoice_is_structurally_broken(world):
    invoices, _, _, anomalies = world
    labelled = {a.entity_id for a in anomalies}
    assert not [i.invoice_id for i in invoices if i.invoice_id not in labelled and _structural_problem(i)]


def test_duplicates_reuse_an_existing_number_from_the_same_supplier(world):
    invoices, _, _, anomalies = world
    dups = {a.entity_id for a in anomalies if a.anomaly_type is AnomalyType.duplicate_invoice}
    originals = {(i.supplier.party_id, i.invoice_number) for i in invoices if i.invoice_id not in dups}
    for i in invoices:
        if i.invoice_id in dups:
            assert (i.supplier.party_id, i.invoice_number) in originals


def test_outliers_stand_far_above_supplier_history(world):
    invoices, _, _, anomalies = world
    labelled = {a.entity_id for a in anomalies}
    history = defaultdict(list)
    for i in invoices:
        if i.invoice_id not in labelled:
            history[i.supplier.party_id].append(i.taxable_total)
    for a in anomalies:
        if a.anomaly_type is AnomalyType.amount_outlier:
            inv = next(i for i in invoices if i.invoice_id == a.entity_id)
            assert inv.taxable_total > 3 * median(history[inv.supplier.party_id])


def test_no_unlabelled_invoice_reaches_outlier_territory(world):
    """A normal invoice above the outlier threshold would be an unlabelled anomaly: label noise."""
    from gstpipe.generator.anomalies import OUTLIER_MULTIPLE

    invoices, _, _, anomalies = world
    labelled = {a.entity_id for a in anomalies}
    history = defaultdict(list)
    for i in invoices:
        if i.invoice_id not in labelled:
            history[i.supplier.party_id].append(i.taxable_total)
    worst = max(max(values) / median(values) for values in history.values())
    assert worst < Decimal(str(OUTLIER_MULTIPLE[0])), f"normal invoice at {worst:.1f}x supplier median"


def test_unbilled_payments_have_no_match(world):
    _, txns, matches, anomalies = world
    matched = {m.txn_id for m in matches}
    unbilled = [a.entity_id for a in anomalies if a.anomaly_type is AnomalyType.unbilled_payment]
    kinds = {t.txn_id: t.kind for t in txns}
    assert all(kinds[t] is TxnKind.unbilled_vendor_payment and t not in matched for t in unbilled)


def test_anomalous_invoices_are_sometimes_paid_and_sometimes_not(world):
    _, _, matches, anomalies = world
    paid = {m.invoice_id for m in matches}
    flagged = [a.entity_id for a in anomalies if a.entity_type == "invoice"]
    assert 0 < sum(f in paid for f in flagged) < len(flagged)


def test_settlement_integrity_survives_injection(world):
    invoices, txns, matches, _ = world
    totals = {i.invoice_id: i.invoice_total for i in invoices}
    amounts = {t.txn_id: t.amount for t in txns}
    by_inv, by_txn = defaultdict(Decimal), defaultdict(Decimal)
    for m in matches:
        by_inv[m.invoice_id] += m.allocated_amount + m.tds_amount
        by_txn[m.txn_id] += m.allocated_amount
    assert all(v == totals[k] for k, v in by_inv.items())
    assert all(v == amounts[k] for k, v in by_txn.items())
    assert len({t.txn_id for t in txns}) == len(txns)
