from decimal import Decimal

import pytest

from gstpipe.render.words import number_in_words, rupees_in_words


@pytest.mark.parametrize("n, expected", [
    (0, "Zero"),
    (7, "Seven"),
    (17, "Seventeen"),
    (80, "Eighty"),
    (105, "One Hundred and Five"),
    (999, "Nine Hundred and Ninety Nine"),
    (1_000, "One Thousand"),
    (21_436, "Twenty One Thousand Four Hundred and Thirty Six"),
    (100_000, "One Lakh"),
    (1_21_567, "One Lakh Twenty One Thousand Five Hundred and Sixty Seven"),
    (10_000_000, "One Crore"),
    (12_34_56_789, "Twelve Crore Thirty Four Lakh Fifty Six Thousand Seven Hundred and Eighty Nine"),
])
def test_indian_grouping(n, expected):
    assert number_in_words(n) == expected


def test_rupees_without_paise():
    assert rupees_in_words(Decimal("21436.00")) == "Twenty One Thousand Four Hundred and Thirty Six Rupees Only"


def test_rupees_with_paise():
    assert rupees_in_words(Decimal("1250.75")) == "One Thousand Two Hundred and Fifty Rupees and Seventy Five Paise Only"


def test_negative_round_off_amount():
    assert rupees_in_words(Decimal("-0.45")).startswith("Minus Zero Rupees and Forty Five Paise")
