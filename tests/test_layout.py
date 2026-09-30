from decimal import Decimal

import pytest

from gstpipe.render.layout import inr


@pytest.mark.parametrize("value, expected", [
    (0, "0.00"),
    (999, "999.00"),
    (1000, "1,000.00"),
    (99999, "99,999.00"),
    (100000, "1,00,000.00"),
    (1234567.5, "12,34,567.50"),
    (Decimal("21436.00"), "21,436.00"),
    (Decimal("123456789.99"), "12,34,56,789.99"),
    (Decimal("-0.45"), "-0.45"),
])
def test_indian_digit_grouping(value, expected):
    assert inr(value) == expected


def test_without_decimals():
    assert inr(Decimal("1234567.89"), decimals=False) == "12,34,567"
