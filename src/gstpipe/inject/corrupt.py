"""Controlled corruption of extracted fields: the instrument of the propagation study.

Random character noise would describe nothing real. These failure modes are the ones tesseract
actually produces on the rendered documents -- confusable glyphs, transposed digits, a displaced
decimal, a truncated name, a dropped field, tax components swapped, a missing line.

The rate is the share of *fields* corrupted, so a sweep over it is a sweep over upstream quality.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass
from datetime import timedelta
from decimal import Decimal

from gstpipe.extract.records import ExtractedInvoice, ExtractedLine

CONFUSABLE = {"0": "O", "O": "0", "1": "l", "l": "1", "5": "S", "S": "5", "8": "B", "B": "8",
              "2": "Z", "Z": "2", "6": "G", "G": "6", "rn": "m", "D": "0"}

TEXT_FIELDS = ("supplier_name", "recipient_name")
CODE_FIELDS = ("supplier_gstin", "recipient_gstin", "invoice_number", "place_of_supply")
MONEY_FIELDS = ("taxable_total", "cgst_total", "sgst_total", "igst_total", "invoice_total")
DATE_FIELDS = ("invoice_date", "due_date")
ALL_FIELDS = TEXT_FIELDS + CODE_FIELDS + MONEY_FIELDS + DATE_FIELDS

MODES = ("confusable", "transpose", "decimal_shift", "truncate", "omit", "tax_swap", "line_drop")


@dataclass
class InjectionReport:
    rate: float
    fields_considered: int = 0
    fields_corrupted: int = 0
    by_mode: dict[str, int] = None

    def __post_init__(self):
        if self.by_mode is None:
            self.by_mode = dict.fromkeys(MODES, 0)

    @property
    def realised_rate(self) -> float:
        return self.fields_corrupted / self.fields_considered if self.fields_considered else 0.0


def _confuse(text: str, rng: random.Random) -> str:
    positions = [i for i, ch in enumerate(text) if ch in CONFUSABLE]
    if not positions:
        return text
    index = rng.choice(positions)
    return text[:index] + CONFUSABLE[text[index]] + text[index + 1:]


def _transpose_digits(value: Decimal, rng: random.Random) -> Decimal:
    digits = re.sub(r"\D", "", f"{value:.2f}")
    if len(digits) < 3:
        return value
    index = rng.randrange(len(digits) - 1)
    swapped = digits[:index] + digits[index + 1] + digits[index] + digits[index + 2:]
    return Decimal(swapped[:-2] + "." + swapped[-2:])


def _shift_decimal(value: Decimal, rng: random.Random) -> Decimal:
    return value * Decimal(rng.choice(("10", "0.1", "100")))


def _corrupt_text(value: str, rng: random.Random) -> tuple[str | None, str]:
    mode = rng.choices(("confusable", "truncate", "omit"), weights=(0.6, 0.3, 0.1))[0]
    if mode == "omit":
        return None, mode
    if mode == "truncate":
        cut = max(3, int(len(value) * rng.uniform(0.4, 0.8)))
        return value[:cut].strip(), mode
    return _confuse(value, rng), mode


def _corrupt_code(value: str, rng: random.Random) -> tuple[str | None, str]:
    mode = rng.choices(("confusable", "omit", "truncate"), weights=(0.75, 0.15, 0.1))[0]
    if mode == "omit":
        return None, mode
    if mode == "truncate":
        return value[:-rng.randint(1, 2)], mode
    return _confuse(value, rng), mode


def _corrupt_money(value: Decimal, rng: random.Random) -> tuple[Decimal | None, str]:
    mode = rng.choices(("transpose", "confusable", "decimal_shift", "omit"),
                       weights=(0.45, 0.2, 0.2, 0.15))[0]
    if mode == "omit":
        return None, mode
    if mode == "transpose":
        return _transpose_digits(value, rng), mode
    if mode == "decimal_shift":
        return _shift_decimal(value, rng), mode
    corrupted = _confuse(f"{value:.2f}", rng)
    digits = re.sub(r"[^\d.]", "", corrupted) or "0"
    try:
        return Decimal(digits), mode
    except Exception:
        return None, "omit"


def corrupt(extracted: ExtractedInvoice, rate: float, rng: random.Random,
            report: InjectionReport | None = None) -> ExtractedInvoice:
    """Return a copy with about `rate` of its populated fields damaged."""
    report = report or InjectionReport(rate)
    updates: dict[str, object] = {}

    for field in ALL_FIELDS:
        value = getattr(extracted, field)
        if value is None:
            continue
        report.fields_considered += 1
        if rng.random() >= rate:
            continue
        report.fields_corrupted += 1

        if field in TEXT_FIELDS:
            new_value, mode = _corrupt_text(str(value), rng)
        elif field in CODE_FIELDS:
            new_value, mode = _corrupt_code(str(value), rng)
        elif field in MONEY_FIELDS:
            new_value, mode = _corrupt_money(Decimal(str(value)), rng)
        else:
            new_value, mode = value + timedelta(days=rng.choice((-30, -1, 1, 10, 30))), "transpose"
        updates[field] = new_value
        report.by_mode[mode] = report.by_mode.get(mode, 0) + 1

    # Tax components swapped between CGST and SGST: a column confusion, not a digit error.
    if extracted.cgst_total is not None and extracted.igst_total is not None and rng.random() < rate / 2:
        updates["cgst_total"], updates["igst_total"] = extracted.igst_total, extracted.cgst_total
        report.by_mode["tax_swap"] += 1

    lines = list(extracted.line_items)
    if lines and rng.random() < rate / 2:
        dropped = rng.randrange(len(lines))
        lines = [li for i, li in enumerate(lines) if i != dropped]
        report.by_mode["line_drop"] += 1
    updates["line_items"] = [_corrupt_line(li, rate, rng) for li in lines]

    return extracted.model_copy(update=updates)


def _corrupt_line(line: ExtractedLine, rate: float, rng: random.Random) -> ExtractedLine:
    updates = {}
    if line.description and rng.random() < rate:
        updates["description"] = _corrupt_text(line.description, rng)[0]
    if line.hsn_sac and rng.random() < rate:
        updates["hsn_sac"] = _corrupt_code(line.hsn_sac, rng)[0]
    if line.taxable_value is not None and rng.random() < rate:
        updates["taxable_value"] = _corrupt_money(line.taxable_value, rng)[0]
    return line.model_copy(update=updates) if updates else line


def from_invoice(invoice) -> ExtractedInvoice:
    """A perfect extraction of an invoice: the clean starting point that injection degrades."""
    return ExtractedInvoice(
        document_id=invoice.invoice_id,
        supplier_name=invoice.supplier.name, supplier_gstin=invoice.supplier.gstin,
        recipient_name=invoice.recipient.name, recipient_gstin=invoice.recipient.gstin,
        invoice_number=invoice.invoice_number, invoice_date=invoice.invoice_date,
        due_date=invoice.due_date, place_of_supply=invoice.place_of_supply,
        taxable_total=invoice.taxable_total, cgst_total=invoice.cgst_total,
        sgst_total=invoice.sgst_total, igst_total=invoice.igst_total,
        invoice_total=invoice.invoice_total, round_off=invoice.round_off,
        line_items=[ExtractedLine(description=li.description, hsn_sac=li.hsn_sac or None,
                                  quantity=li.quantity, unit_price=li.unit_price,
                                  taxable_value=li.taxable_value, gst_rate=li.gst_rate)
                    for li in invoice.line_items],
    )
