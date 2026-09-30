"""Tier C1: a keyword and HSN lookup, the way a hand-built classifier in a finance team works."""

from __future__ import annotations

import re

from gstpipe import catalog
from gstpipe.categorise.features import CategoryInput

UNKNOWN = "Unknown"

# Checked in order: the first category with a keyword hit wins.
KEYWORDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("cloud_services", ("cloud", "hosting", "object storage", "managed database")),
    ("software_licences", ("licence", "license", "subscription", "software")),
    ("it_consulting", ("it support", "retainer", "website development", "network setup")),
    ("it_hardware", ("laptop", "monitor", "keyboard", "mouse", "desktop")),
    ("furniture", ("chair", "desk", "workstation", "cabinet", "furniture")),
    ("rent", ("rent", "lease", "premises", "warehouse lease")),
    ("telecom", ("internet", "broadband", "leased line", "postpaid", "telecom")),
    ("repairs", ("repair", "servicing", "maintenance", "electrical", "cement")),
    ("legal", ("legal", "contract drafting", "trademark", "advisory")),
    ("accounting", ("audit", "bookkeeping", "gst return", "accounting")),
    ("advertising", ("advertis", "ad campaign", "hoarding", "promotion", "digital ad")),
    ("hotel", ("hotel", "room tariff", "accommodation", "stay")),
    ("air_travel", ("air ticket", "flight", "airfare", "economy class")),
    ("freight", ("freight", "transport", "delivery", "courier")),
    ("raw_materials", ("steel", "granules", "yarn", "coil", "hdpe")),
    ("packaging", ("carton", "stretch wrap", "bubble wrap", "packaging")),
    ("office_supplies", ("paper", "toner", "stationery", "copier")),
)

HSN_TO_CATEGORY = {spec.hsn_sac: spec.category for spec in catalog.ITEMS}


def predict(item: CategoryInput) -> str:
    """HSN/SAC is checked first because a code is unambiguous when it is present and readable."""
    for code in item.hsn_codes:
        if (key := HSN_TO_CATEGORY.get(code.strip())) is not None:
            return catalog.category_path(key)

    text = item.text
    for key, words in KEYWORDS:
        if any(re.search(re.escape(word), text) for word in words):
            return catalog.category_path(key)
    return UNKNOWN
