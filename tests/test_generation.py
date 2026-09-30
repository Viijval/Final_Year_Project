import random
from collections import Counter
from datetime import date
from decimal import Decimal

import pytest

from gstpipe.generator.entities import make_business
from gstpipe.generator.invoices import make_invoices
from gstpipe.gst import is_valid_gstin, money, valid_slabs

START, END = date(2025, 4, 1), date(2026, 3, 31)


@pytest.fixture(scope="module")
def world():
    rng = random.Random(42)
    used: set[str] = set()
    splits = ["train"] * 6 + ["val"] * 2 + ["test"] * 2
    businesses = [make_business(rng, f"B{i:03d}", s, used) for i, s in enumerate(splits)]
    invoices = [inv for b in businesses for inv in make_invoices(rng, b, START, END)]
    return businesses, invoices


def test_vendor_names_never_cross_splits(world):
    businesses, _ = world
    names_by_split: dict[str, set[str]] = {}
    for b in businesses:
        names_by_split.setdefault(b.split, set()).update(v.party.name for v in b.vendors)
    assert not names_by_split["train"] & names_by_split["test"]
    assert not names_by_split["train"] & names_by_split["val"]
    assert not names_by_split["val"] & names_by_split["test"]


def test_every_party_has_valid_gstin_matching_its_state(world):
    businesses, _ = world
    for party in [b.party for b in businesses] + [v.party for b in businesses for v in b.vendors]:
        assert is_valid_gstin(party.gstin), party.gstin
        assert party.gstin[:2] == party.state_code
        assert party.gstin[2:12] == party.pan


def test_line_and_invoice_arithmetic_is_exact(world):
    _, invoices = world
    for inv in invoices:
        for li in inv.line_items:
            assert li.taxable_value == money(li.quantity * li.unit_price)
        assert inv.taxable_total == sum(li.taxable_value for li in inv.line_items)
        gross = inv.taxable_total + inv.cgst_total + inv.sgst_total + inv.igst_total
        assert inv.invoice_total == gross + inv.round_off
        assert abs(inv.round_off) <= Decimal("0.50")
        assert inv.invoice_total == inv.invoice_total.to_integral_value()


def test_tax_type_follows_place_of_supply(world):
    _, invoices = world
    for inv in invoices:
        intra = inv.supplier.state_code == inv.place_of_supply
        for li in inv.line_items:
            if intra:
                assert li.igst == 0 and li.cgst == li.sgst
            else:
                assert li.cgst == 0 and li.sgst == 0


def test_rates_belong_to_the_slab_schedule_of_the_invoice_date(world):
    _, invoices = world
    for inv in invoices:
        assert all(li.gst_rate in valid_slabs(inv.invoice_date) for li in inv.line_items)


def test_both_slab_regimes_and_multiline_invoices_occur(world):
    _, invoices = world
    assert any(inv.invoice_date < date(2025, 9, 22) for inv in invoices)
    assert any(inv.invoice_date >= date(2025, 9, 22) for inv in invoices)
    assert Counter(len(inv.line_items) for inv in invoices)[1] < len(invoices)


def test_rent_bills_roughly_monthly(world):
    businesses, invoices = world
    per_vendor = Counter(inv.supplier.party_id for inv in invoices)
    rent_vendors = [v.party.party_id for b in businesses for v in b.vendors if v.category == "rent"]
    assert rent_vendors
    for vid in rent_vendors:
        assert 8 <= per_vendor[vid] <= 30


def test_invoice_numbers_unique_per_vendor(world):
    _, invoices = world
    seen = Counter((inv.supplier.party_id, inv.invoice_number) for inv in invoices)
    assert max(seen.values()) == 1


def test_same_seed_reproduces_identical_data():
    def run():
        rng = random.Random(9)
        b = make_business(rng, "B000", "train", set())
        return [inv.model_dump_json() for inv in make_invoices(rng, b, START, END)]
    assert run() == run()
