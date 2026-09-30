"""Purchase invoices with correct GST arithmetic, generated as realistic monthly billing histories."""

from __future__ import annotations

import random
import re
from calendar import monthrange
from datetime import date, timedelta
from decimal import ROUND_HALF_UP, Decimal

from gstpipe import catalog
from gstpipe.generator.entities import Business, Vendor
from gstpipe.gst import SLAB_CHANGE, compute_tax, money
from gstpipe.schema import Invoice, LineItem

# Expected invoices per vendor per month.
MONTHLY_RATE: dict[str, float] = {
    "rent": 1.0, "telecom": 0.95, "cloud_services": 0.95, "accounting": 0.8, "raw_materials": 1.8,
    "freight": 1.4, "packaging": 0.8, "office_supplies": 0.5, "software_licences": 0.35, "advertising": 0.35,
    "air_travel": 0.35, "hotel": 0.3, "it_consulting": 0.3, "repairs": 0.25, "legal": 0.15,
    "it_hardware": 0.15, "furniture": 0.08,
}
PAYMENT_TERMS_DAYS = (0, 7, 15, 30, 30, 45)


def _months(start: date, end: date):
    y, m = start.year, start.month
    while (y, m) <= (end.year, end.month):
        yield y, m
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)


def _fy(on: date) -> str:
    first = on.year if on.month >= 4 else on.year - 1
    return f"{first % 100:02d}-{(first + 1) % 100:02d}"


class _NumberSeries:
    """Per-vendor invoice numbering in one of the formats Indian suppliers commonly use."""

    def __init__(self, rng: random.Random, vendor_name: str):
        self.style = rng.randrange(3)
        self.seq = rng.randint(1, 900)
        self.initials = "".join(w[0] for w in vendor_name.split() if w[0].isalpha())[:4].upper() or "INV"

    def next(self, on: date) -> str:
        self.seq += 1
        if self.style == 0:
            return f"INV-{self.seq:04d}"
        if self.style == 1:
            return f"{self.initials}/{_fy(on)}/{self.seq:04d}"
        return f"{self.initials}{on.year % 100:02d}{self.seq:05d}"


def _hsn_policy(vendor: Vendor, spec: catalog.ItemSpec) -> tuple[str, bool]:
    """A vendor uses one code consistently, and some vendors never print HSN at all."""
    rng = random.Random(f"hsn:{vendor.party.party_id}")
    return rng.choice(spec.hsn_codes), rng.random() < 0.22


def _hsn_for_line(rng: random.Random, vendor: Vendor, spec: catalog.ItemSpec) -> str:
    code, omits = _hsn_policy(vendor, spec)
    if omits:
        return ""
    if rng.random() < 0.03:  # occasional wrong code, as happens in practice
        return rng.choice(catalog.ALL_HSN_CODES)
    return code


def _describe(rng: random.Random, base: str, on: date) -> str:
    """How a supplier actually writes the line: references, periods and abbreviations."""
    text = rng.choice(catalog.DESCRIPTION_PREFIXES) + base
    text += rng.choice(catalog.DESCRIPTION_SUFFIXES).format(
        month=on.strftime("%b"), yy=on.strftime("%y"), po=rng.randrange(100_000, 999_999))
    if rng.random() < 0.25:
        for long_form, short in catalog.ABBREVIATIONS.items():
            text = re.sub(long_form, short, text, flags=re.IGNORECASE)
    return (text.upper() if rng.random() < 0.12 else text).strip()


def _line(rng: random.Random, vendor: Vendor, spec: catalog.ItemSpec, line_no: int, description: str,
          on: date, intra: bool) -> LineItem:
    lo, hi = spec.unit_price
    centre = lo + (hi - lo) * vendor.price_scale
    unit_price = money(min(hi, max(lo, rng.gauss(centre, (hi - lo) * 0.08))))
    q_lo, q_hi = spec.quantity
    q_centre = q_lo + (q_hi - q_lo) * vendor.qty_scale
    quantity = Decimal(min(q_hi, max(q_lo, round(rng.gauss(q_centre, (q_hi - q_lo) * 0.1)))))
    taxable = money(unit_price * quantity)
    rate = spec.rate_after if on >= SLAB_CHANGE else spec.rate_before
    tax = compute_tax(taxable, rate, intra)
    return LineItem(line_no=line_no, description=_describe(rng, description, on),
                    hsn_sac=_hsn_for_line(rng, vendor, spec),
                    quantity=quantity, unit_price=unit_price, taxable_value=taxable, gst_rate=rate,
                    cgst=tax.cgst, sgst=tax.sgst, igst=tax.igst, category=catalog.category_path(vendor.category))


def build_invoice(rng: random.Random, business: Business, vendor: Vendor, invoice_id: str,
                  number: str, on: date) -> Invoice:
    intra = vendor.party.state_code == business.party.state_code
    spec = rng.choice(catalog.ITEMS_BY_CATEGORY[vendor.category])
    descriptions = rng.sample(spec.descriptions, k=rng.choice((1, 1, 1, 2, 2, 3)))
    lines = [_line(rng, vendor, spec, i + 1, desc, on, intra) for i, desc in enumerate(descriptions)]
    return finalise(Invoice(
        invoice_id=invoice_id, business_id=business.party.party_id, split=business.split, invoice_number=number,
        invoice_date=on, due_date=on + timedelta(days=rng.choice(PAYMENT_TERMS_DAYS)), supplier=vendor.party,
        recipient=business.party, place_of_supply=business.party.state_code, line_items=lines,
        taxable_total=money(0), cgst_total=money(0), sgst_total=money(0), igst_total=money(0),
        round_off=money(0), invoice_total=money(0), category=lines[0].category,
    ))


def finalise(invoice: Invoice) -> Invoice:
    """Recompute totals, rupee round-off and primary category from the line items."""
    lines = invoice.line_items
    taxable = sum((li.taxable_value for li in lines), money(0))
    cgst = sum((li.cgst for li in lines), money(0))
    sgst = sum((li.sgst for li in lines), money(0))
    igst = sum((li.igst for li in lines), money(0))
    gross = taxable + cgst + sgst + igst
    rounded = gross.quantize(Decimal("1"), rounding=ROUND_HALF_UP)
    primary = max(lines, key=lambda li: li.taxable_value).category
    return invoice.model_copy(update={
        "taxable_total": taxable, "cgst_total": cgst, "sgst_total": sgst, "igst_total": igst,
        "round_off": money(rounded - gross), "invoice_total": money(rounded), "category": primary,
    })


def make_invoices(rng: random.Random, business: Business, start: date, end: date) -> list[Invoice]:
    series = {v.party.party_id: _NumberSeries(rng, v.party.name) for v in business.vendors}
    invoices: list[Invoice] = []
    for year, month in _months(start, end):
        first = max(start, date(year, month, 1))
        last = min(end, date(year, month, monthrange(year, month)[1]))
        span = (last - first).days
        for vendor in business.vendors:
            # Whole part bills every month (rent is always monthly); the fraction adds an occasional extra bill.
            rate = MONTHLY_RATE[vendor.category]
            count = int(rate) + (rng.random() < rate - int(rate))
            for on in sorted(first + timedelta(days=rng.randint(0, span)) for _ in range(count)):
                inv_id = f"{business.party.party_id}-I{len(invoices) + 1:05d}"
                invoices.append(build_invoice(rng, business, vendor, inv_id, series[vendor.party.party_id].next(on), on))
    invoices.sort(key=lambda inv: (inv.invoice_date, inv.invoice_id))
    return invoices
