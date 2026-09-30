"""Businesses (invoice recipients) and their vendors (suppliers).

Every generated name is unique across the whole run, so no vendor can appear in more than one split.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from gstpipe import catalog
from gstpipe.gst import STATES, make_gstin, make_pan
from gstpipe.schema import LegalForm, Party, Split

SECTORS: dict[str, tuple[str, ...]] = {
    "manufacturing": ("raw_materials", "packaging", "freight", "repairs", "rent", "telecom",
                      "office_supplies", "accounting", "legal"),
    "it_services": ("cloud_services", "software_licences", "it_hardware", "furniture", "rent", "telecom",
                    "legal", "accounting", "advertising", "hotel", "air_travel"),
    "trading": ("packaging", "freight", "rent", "telecom", "accounting", "advertising", "office_supplies",
                "it_hardware"),
    "agency": ("software_licences", "it_consulting", "it_hardware", "rent", "telecom", "hotel", "air_travel",
               "legal", "accounting", "advertising"),
}

INTRA_STATE_SHARE = 0.7


@dataclass
class Vendor:
    party: Party
    category: str
    price_scale: float  # where in the item's price range this vendor typically bills, 0..1

    @property
    def qty_scale(self) -> float:
        """Typical order size within the item's quantity range; pricier vendors sell in smaller lots."""
        return 1.0 - self.price_scale


@dataclass
class Business:
    party: Party
    sector: str
    split: Split
    vendors: list[Vendor]


def _party(rng: random.Random, party_id: str, name: str, legal_form: LegalForm, state_code: str) -> Party:
    pan = make_pan(rng, legal_form.value, name)
    city = rng.choice(STATES[state_code][1])
    address = f"{rng.randint(1, 480)}, {rng.choice(catalog.STREETS)}, {city}, {STATES[state_code][0]}"
    return Party(party_id=party_id, name=name, legal_form=legal_form, pan=pan,
                 gstin=make_gstin(rng, state_code, pan), state_code=state_code, city=city, address=address)


def _vendor_name(rng: random.Random, category: str, legal_form: LegalForm, used: set[str]) -> str:
    suffix = catalog.LEGAL_SUFFIX[legal_form.value]
    for _ in range(500):
        if rng.random() < 0.4:
            core = rng.choice(catalog.INFORMATIVE_CORES[category])
        else:
            lead = rng.choice(catalog.GENERIC_PREFIXES + catalog.SURNAMES)
            middle = rng.choice(catalog.GENERIC_MIDDLES)
            core = " ".join(part for part in (lead, middle, rng.choice(catalog.GENERIC_CORES)) if part)
        if (name := core + suffix) not in used:
            used.add(name)
            return name
    raise RuntimeError(f"vendor name space exhausted for {category!r}; generate fewer businesses")


def make_business(rng: random.Random, business_id: str, split: Split, used_names: set[str],
                  vendors_per_category: tuple[int, int] = (1, 3)) -> Business:
    sector = rng.choice(tuple(SECTORS))
    state = rng.choice(tuple(STATES))
    form = rng.choice((LegalForm.company, LegalForm.llp, LegalForm.firm))
    base = f"{rng.choice(catalog.SURNAMES)} {sector.replace('_', ' ').title()}"
    name = base + catalog.LEGAL_SUFFIX[form.value]
    while name in used_names:
        name = f"{rng.choice(catalog.GENERIC_PREFIXES)} {base}{catalog.LEGAL_SUFFIX[form.value]}"
        base = name.removesuffix(catalog.LEGAL_SUFFIX[form.value]) if catalog.LEGAL_SUFFIX[form.value] else name
    used_names.add(name)

    vendors: list[Vendor] = []
    for category in SECTORS[sector]:
        for _ in range(rng.randint(*vendors_per_category)):
            v_state = state if rng.random() < INTRA_STATE_SHARE else rng.choice([s for s in STATES if s != state])
            v_form = rng.choice(tuple(LegalForm))
            v_party = _party(rng, f"{business_id}-V{len(vendors) + 1:03d}",
                             _vendor_name(rng, category, v_form, used_names), v_form, v_state)
            vendors.append(Vendor(party=v_party, category=category, price_scale=rng.uniform(0.15, 0.85)))
    return Business(party=_party(rng, business_id, name, form, state), sector=sector, split=split, vendors=vendors)
