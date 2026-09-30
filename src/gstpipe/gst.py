"""GST domain rules: GSTIN/PAN structure, state codes, rate slabs and tax computation.

All money is handled as Decimal and rounded half-up to paise, matching how invoices are printed.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import date
from decimal import ROUND_HALF_UP, Decimal

_CODEPOINTS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
PAISE = Decimal("0.01")

# GST state codes (first two digits of a GSTIN) with a few representative cities each.
STATES: dict[str, tuple[str, tuple[str, ...]]] = {
    "07": ("Delhi", ("New Delhi", "Dwarka", "Rohini")),
    "06": ("Haryana", ("Gurugram", "Faridabad", "Panipat")),
    "08": ("Rajasthan", ("Jaipur", "Jodhpur", "Udaipur")),
    "09": ("Uttar Pradesh", ("Noida", "Lucknow", "Kanpur")),
    "19": ("West Bengal", ("Kolkata", "Howrah", "Durgapur")),
    "24": ("Gujarat", ("Ahmedabad", "Surat", "Vadodara")),
    "27": ("Maharashtra", ("Mumbai", "Pune", "Nagpur")),
    "29": ("Karnataka", ("Bengaluru", "Mysuru", "Hubballi")),
    "32": ("Kerala", ("Kochi", "Thiruvananthapuram", "Kozhikode")),
    "33": ("Tamil Nadu", ("Chennai", "Coimbatore", "Madurai")),
    "36": ("Telangana", ("Hyderabad", "Warangal", "Karimnagar")),
}

# Fourth PAN character encodes the holder type.
PAN_HOLDER_TYPE = {"proprietorship": "P", "company": "C", "firm": "F", "llp": "F"}

# Slab rationalisation approved by the GST Council, effective 22 Sep 2025 (5% / 18% / 40%).
# Verify against current CBIC notifications before relying on specific slabs.
SLAB_CHANGE = date(2025, 9, 22)
SLABS_BEFORE = (Decimal("0"), Decimal("5"), Decimal("12"), Decimal("18"), Decimal("28"))
SLABS_AFTER = (Decimal("0"), Decimal("5"), Decimal("18"), Decimal("40"))


def money(value: Decimal | int | float | str) -> Decimal:
    return Decimal(str(value)).quantize(PAISE, rounding=ROUND_HALF_UP)


def valid_slabs(on: date) -> tuple[Decimal, ...]:
    return SLABS_AFTER if on >= SLAB_CHANGE else SLABS_BEFORE


def gstin_check_char(first14: str) -> str:
    """Luhn mod-36 check character over the first 14 GSTIN characters."""
    factor, total, mod = 2, 0, len(_CODEPOINTS)
    for ch in reversed(first14.upper()):
        product = factor * _CODEPOINTS.index(ch)
        factor = 1 if factor == 2 else 2
        total += product // mod + product % mod
    return _CODEPOINTS[(mod - total % mod) % mod]


def is_valid_gstin(gstin: str) -> bool:
    g = gstin.strip().upper()
    if len(g) != 15 or g[:2] not in STATES or not g[2:7].isalpha() or not g[7:11].isdigit():
        return False
    if not g[11].isalpha() or g[13] != "Z" or any(c not in _CODEPOINTS for c in g):
        return False
    return gstin_check_char(g[:14]) == g[14]


def make_pan(rng: random.Random, legal_form: str, name: str) -> str:
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    first_letter = next((c for c in name.upper() if c.isalpha()), "X")
    return (
        "".join(rng.choice(letters) for _ in range(3))
        + PAN_HOLDER_TYPE[legal_form]
        + first_letter
        + f"{rng.randint(0, 9999):04d}"
        + rng.choice(letters)
    )


def make_gstin(rng: random.Random, state_code: str, pan: str, entity_number: int = 1) -> str:
    body = f"{state_code}{pan}{_CODEPOINTS[entity_number]}Z"
    return body + gstin_check_char(body)


@dataclass(frozen=True)
class TaxSplit:
    cgst: Decimal
    sgst: Decimal
    igst: Decimal

    @property
    def total(self) -> Decimal:
        return self.cgst + self.sgst + self.igst


def compute_tax(taxable: Decimal, rate: Decimal, intra_state: bool) -> TaxSplit:
    """Intra-state supplies split the rate across CGST and SGST; inter-state supplies charge IGST."""
    zero = money(0)
    if intra_state:
        half = money(taxable * rate / 200)
        return TaxSplit(cgst=half, sgst=half, igst=zero)
    return TaxSplit(cgst=zero, sgst=zero, igst=money(taxable * rate / 100))
