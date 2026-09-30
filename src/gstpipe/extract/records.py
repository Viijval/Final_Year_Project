"""What an extractor returns: the same fields as the ground truth, but every one may be wrong or missing."""

from __future__ import annotations

from datetime import date
from decimal import Decimal

from pydantic import BaseModel, Field

FIELDS = ("supplier_name", "supplier_gstin", "recipient_name", "recipient_gstin", "invoice_number",
          "invoice_date", "due_date", "place_of_supply", "taxable_total", "cgst_total", "sgst_total",
          "igst_total", "invoice_total")
TEXT_FIELDS = frozenset({"supplier_name", "recipient_name"})
MONEY_FIELDS = frozenset({"taxable_total", "cgst_total", "sgst_total", "igst_total", "invoice_total"})


class Region(BaseModel):
    """Where on the page a value was read from, so a person can check it against the scan."""

    x0: float
    y0: float
    x1: float
    y1: float
    text: str = ""
    page_width: float = 1.0
    page_height: float = 1.0

    @property
    def box(self) -> dict[str, float]:
        """Position as percentages, so it overlays an image at any display size."""
        w = self.page_width or 1.0
        h = self.page_height or 1.0
        return {"left": 100 * self.x0 / w, "top": 100 * self.y0 / h,
                "width": 100 * (self.x1 - self.x0) / w, "height": 100 * (self.y1 - self.y0) / h}


class ExtractedLine(BaseModel):
    description: str | None = None
    hsn_sac: str | None = None
    quantity: Decimal | None = None
    unit_price: Decimal | None = None
    taxable_value: Decimal | None = None
    gst_rate: Decimal | None = None
    region: "Region | None" = None


class ExtractedInvoice(BaseModel):
    """Every field optional: a missed field is a different failure from a misread one."""

    document_id: str
    supplier_name: str | None = None
    supplier_gstin: str | None = None
    recipient_name: str | None = None
    recipient_gstin: str | None = None
    invoice_number: str | None = None
    invoice_date: date | None = None
    due_date: date | None = None
    place_of_supply: str | None = None
    taxable_total: Decimal | None = None
    cgst_total: Decimal | None = None
    sgst_total: Decimal | None = None
    igst_total: Decimal | None = None
    invoice_total: Decimal | None = None
    round_off: Decimal | None = None  # not scored in FIELDS; needed for consistency checks
    line_items: list[ExtractedLine] = Field(default_factory=list)
    confidence: dict[str, float] = Field(default_factory=dict)
    provenance: dict[str, Region] = Field(
        default_factory=dict, description="field name -> the part of the page it was read from")

    def present(self) -> set[str]:
        return {f for f in FIELDS if getattr(self, f) is not None}
