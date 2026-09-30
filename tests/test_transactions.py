import random
from collections import Counter, defaultdict
from datetime import date
from decimal import Decimal

import pytest

from gstpipe.generator.entities import make_business
from gstpipe.generator.invoices import make_invoices
from gstpipe.generator.transactions import _KEY_BY_PATH, _TDS_RATE, bank_name, make_transactions
from gstpipe.gst import money
from gstpipe.schema import MatchType, TxnKind

START, END = date(2025, 4, 1), date(2026, 3, 31)


@pytest.fixture(scope="module")
def world():
    rng = random.Random(11)
    used: set[str] = set()
    invoices, txns, matches = [], [], []
    for i in range(8):
        b = make_business(rng, f"B{i:03d}", "train", used)
        inv = make_invoices(rng, b, START, END)
        t, m = make_transactions(rng, b, inv, START, END)
        invoices += inv; txns += t; matches += m
    return {i.invoice_id: i for i in invoices}, {t.txn_id: t for t in txns}, matches


def test_only_invoice_payments_carry_matches(world):
    _, txns, matches = world
    matched = {m.txn_id for m in matches}
    for t in txns.values():
        assert (t.txn_id in matched) == (t.kind is TxnKind.invoice_payment)


def test_allocations_sum_to_transaction_amount(world):
    _, txns, matches = world
    by_txn = defaultdict(Decimal)
    for m in matches:
        by_txn[m.txn_id] += m.allocated_amount
    for txn_id, total in by_txn.items():
        assert total == txns[txn_id].amount


def test_every_matched_invoice_is_fully_settled(world):
    invoices, _, matches = world
    covered = defaultdict(Decimal)
    for m in matches:
        covered[m.invoice_id] += m.allocated_amount + m.tds_amount
    for inv_id, total in covered.items():
        assert total == invoices[inv_id].invoice_total


def test_payments_never_precede_invoice_and_stay_in_window(world):
    invoices, txns, matches = world
    for m in matches:
        assert invoices[m.invoice_id].invoice_date <= txns[m.txn_id].txn_date <= END
    assert all(START <= t.txn_date <= END for t in txns.values())


def test_tds_only_on_applicable_categories_at_statutory_rate(world):
    invoices, _, matches = world
    for m in matches:
        inv = invoices[m.invoice_id]
        rate = _TDS_RATE[_KEY_BY_PATH[inv.category]]
        if m.tds_amount:
            assert rate > 0 and m.match_type is MatchType.net_of_tds
            assert m.tds_amount == money(inv.taxable_total * rate / 100)


def test_consolidated_payments_cover_several_invoices_of_one_supplier(world):
    invoices, _, matches = world
    groups = defaultdict(list)
    for m in matches:
        if m.match_type is MatchType.consolidated:
            groups[m.txn_id].append(invoices[m.invoice_id].supplier.party_id)
    assert groups
    assert all(len(sup) >= 2 and len(set(sup)) == 1 for sup in groups.values())


def test_all_match_types_and_decoy_kinds_occur(world):
    _, txns, matches = world
    assert set(Counter(m.match_type for m in matches)) == set(MatchType)
    kinds = {t.kind for t in txns.values()}
    assert {TxnKind.salary, TxnKind.gst_remittance, TxnKind.bank_charge} <= kinds


def test_some_invoices_remain_unpaid(world):
    invoices, _, matches = world
    assert len(invoices) > len({m.invoice_id for m in matches})


@pytest.mark.parametrize("raw, limit, expected", [
    ("Nimbus Cloud Pvt Ltd", 22, "NIMBUS CLOUD PVT"),
    ("Iyer & Rao Law LLP", 30, "IYER AND RAO LAW LLP"),
    ("Shree Balaji Global Enterprises & Co.", 18, "SHREE BALAJI GLOBA"),
])
def test_bank_name_normalises_and_truncates(raw, limit, expected):
    assert bank_name(raw, limit) == expected


def test_same_seed_reproduces_identical_transactions():
    def run():
        rng = random.Random(5)
        b = make_business(rng, "B000", "train", set())
        t, m = make_transactions(rng, b, make_invoices(rng, b, START, END), START, END)
        return [x.model_dump_json() for x in t + m]
    assert run() == run()
