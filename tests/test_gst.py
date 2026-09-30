import random
from datetime import date
from decimal import Decimal

import pytest

from gstpipe.gst import (
    SLABS_AFTER,
    SLABS_BEFORE,
    compute_tax,
    gstin_check_char,
    is_valid_gstin,
    make_gstin,
    make_pan,
    money,
    valid_slabs,
)


def test_known_published_sample_gstin_validates():
    assert is_valid_gstin("27AAPFU0939F1ZV")


def test_generated_gstins_always_validate():
    rng = random.Random(7)
    for _ in range(500):
        pan = make_pan(rng, "company", "Apex Traders")
        assert is_valid_gstin(make_gstin(rng, "29", pan))


def test_single_character_corruption_breaks_checksum():
    gstin = make_gstin(random.Random(1), "27", "AAPCA1234B")
    corrupted = gstin[:9] + ("0" if gstin[9] != "0" else "1") + gstin[10:]
    assert not is_valid_gstin(corrupted)


@pytest.mark.parametrize("bad", ["27AAPFU0939F1Z", "99AAPFU0939F1ZV", "27AAPFU0939F1XV", ""])
def test_malformed_gstins_rejected(bad):
    assert not is_valid_gstin(bad)


def test_pan_encodes_holder_type_and_name_initial():
    pan = make_pan(random.Random(3), "llp", "Nimbus Cloud")
    assert len(pan) == 10 and pan[3] == "F" and pan[4] == "N"


def test_intra_state_splits_rate_evenly():
    tax = compute_tax(Decimal("100000"), Decimal("18"), intra_state=True)
    assert (tax.cgst, tax.sgst, tax.igst) == (money(9000), money(9000), money(0))
    assert tax.total == money(18000)


def test_inter_state_charges_igst_only():
    tax = compute_tax(Decimal("100000"), Decimal("18"), intra_state=False)
    assert (tax.cgst, tax.sgst, tax.igst) == (money(0), money(0), money(18000))


def test_odd_paise_round_half_up_per_component():
    tax = compute_tax(Decimal("333.33"), Decimal("5"), intra_state=True)
    assert tax.cgst == tax.sgst == Decimal("8.33")


def test_slab_schedule_switches_on_rationalisation_date():
    assert valid_slabs(date(2025, 9, 21)) == SLABS_BEFORE
    assert valid_slabs(date(2025, 9, 22)) == SLABS_AFTER


def test_check_char_is_uppercase_codepoint():
    assert gstin_check_char("27aapfu0939f1z") == "V"
