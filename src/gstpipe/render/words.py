"""Amount in words using the Indian numbering system, as printed on GST invoices."""

from __future__ import annotations

from decimal import Decimal

_UNITS = ("", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine", "Ten",
          "Eleven", "Twelve", "Thirteen", "Fourteen", "Fifteen", "Sixteen", "Seventeen",
          "Eighteen", "Nineteen")
_TENS = ("", "", "Twenty", "Thirty", "Forty", "Fifty", "Sixty", "Seventy", "Eighty", "Ninety")
# Indian grouping: crore, lakh, thousand, hundred
_SCALES = ((10_000_000, "Crore"), (100_000, "Lakh"), (1_000, "Thousand"), (100, "Hundred"))


def _under_hundred(n: int) -> list[str]:
    if n < 20:
        return [_UNITS[n]] if n else []
    return [_TENS[n // 10]] + ([_UNITS[n % 10]] if n % 10 else [])


def number_in_words(n: int) -> str:
    if n == 0:
        return "Zero"
    if n < 0:
        return "Minus " + number_in_words(-n)
    parts: list[str] = []
    for value, label in _SCALES:
        if n >= value:
            parts += [number_in_words(n // value), label]
            n %= value
    if n:
        if parts:
            parts.append("and")
        parts += _under_hundred(n)
    return " ".join(parts)


def rupees_in_words(amount: Decimal) -> str:
    """e.g. Decimal('121567.50') -> 'One Lakh Twenty One Thousand Five Hundred and Sixty Seven Rupees and Fifty Paise Only'."""
    sign = "Minus " if amount < 0 else ""
    amount = abs(amount)
    rupees = int(amount)
    paise = int((amount - rupees) * 100)
    text = f"{sign}{number_in_words(rupees)} Rupees"
    if paise:
        text += f" and {number_in_words(paise)} Paise"
    return text + " Only"
