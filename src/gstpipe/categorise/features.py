"""The input a categoriser sees.

Built either from ground truth or from Stage 1 output, so the same models can be scored on clean
fields and on extracted fields. Under extraction noise the supplier name and line descriptions
arrive corrupted, which is exactly the effect the propagation study measures.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal

from gstpipe.extract.records import ExtractedInvoice
from gstpipe.schema import Invoice


@dataclass
class CategoryInput:
    invoice_id: str
    supplier_name: str = ""
    descriptions: list[str] = field(default_factory=list)
    hsn_codes: list[str] = field(default_factory=list)
    amount: Decimal | None = None
    invoice_date: date | None = None
    n_lines: int = 0

    @property
    def text(self) -> str:
        """Everything a text model reads: supplier, line descriptions and HSN/SAC codes."""
        parts = [self.supplier_name, *self.descriptions, *self.hsn_codes]
        return re.sub(r"\s+", " ", " ".join(p for p in parts if p)).strip().lower()

    @property
    def numeric(self) -> dict[str, float]:
        """Signals that survive OCR corruption of the text, acting as a redundancy channel."""
        amount = float(self.amount) if self.amount is not None else 0.0
        return {
            # Extraction can return a negative total (a stray minus sign, or a round-off line read
            # as the total), which is meaningless for an invoice and would break the logarithm.
            "log_amount": math.log10(max(amount, 0.0) + 1),
            "n_lines": float(self.n_lines),
            "day_of_month": float(self.invoice_date.day) if self.invoice_date else 0.0,
            "month": float(self.invoice_date.month) if self.invoice_date else 0.0,
            "is_round_1000": float(amount > 0 and amount % 1000 == 0),
            "has_hsn": float(bool(self.hsn_codes)),
        }


def from_invoice(invoice: Invoice) -> CategoryInput:
    return CategoryInput(
        invoice_id=invoice.invoice_id,
        supplier_name=invoice.supplier.name,
        descriptions=[li.description for li in invoice.line_items],
        hsn_codes=[li.hsn_sac for li in invoice.line_items],
        amount=invoice.invoice_total,
        invoice_date=invoice.invoice_date,
        n_lines=len(invoice.line_items),
    )


def from_extracted(extracted: ExtractedInvoice) -> CategoryInput:
    return CategoryInput(
        invoice_id=extracted.document_id,
        supplier_name=extracted.supplier_name or "",
        descriptions=[li.description for li in extracted.line_items if li.description],
        hsn_codes=[li.hsn_sac for li in extracted.line_items if li.hsn_sac],
        amount=extracted.invoice_total,
        invoice_date=extracted.invoice_date,
        n_lines=len(extracted.line_items),
    )
