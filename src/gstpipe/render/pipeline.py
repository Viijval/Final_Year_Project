"""Render invoices to document images: PDF -> raster -> scan degradation."""

from __future__ import annotations

import io
import random
from dataclasses import dataclass
from pathlib import Path

import pymupdf
from PIL import Image
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen.canvas import Canvas

from gstpipe.render.degrade import Degradation, degrade
from gstpipe.render.layout import Pen
from gstpipe.render.templates import TEMPLATES
from gstpipe.schema import Invoice

TEMPLATE_NAMES = tuple(TEMPLATES)


@dataclass
class RenderedDoc:
    invoice_id: str
    business_id: str
    split: str
    template: str
    path: str
    width: int
    height: int
    degradation: Degradation

    def as_row(self) -> dict[str, object]:
        return {"invoice_id": self.invoice_id, "business_id": self.business_id, "split": self.split,
                "template": self.template, "file": self.path, "width": self.width, "height": self.height,
                **self.degradation.as_row()}


def template_for_supplier(supplier_id: str, seed: int = 0) -> str:
    """A supplier always issues invoices on the same layout, the way a real vendor would."""
    return TEMPLATE_NAMES[random.Random(f"{seed}:{supplier_id}").randrange(len(TEMPLATE_NAMES))]


def to_pdf(invoice: Invoice, template: str) -> bytes:
    buffer = io.BytesIO()
    canvas = Canvas(buffer, pagesize=A4)
    canvas.setTitle(invoice.invoice_number)
    TEMPLATES[template](Pen(canvas), invoice)
    canvas.showPage()
    canvas.save()
    return buffer.getvalue()


def rasterise(pdf_bytes: bytes, dpi: int) -> Image.Image:
    with pymupdf.open(stream=pdf_bytes, filetype="pdf") as doc:
        pixmap = doc[0].get_pixmap(dpi=dpi)
    return Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)


def render_invoice(invoice: Invoice, *, dpi: int, severity: tuple[float, float], seed: int,
                   template: str | None = None) -> tuple[Image.Image, str, Degradation]:
    rng = random.Random(f"{seed}:{invoice.invoice_id}")
    name = template or template_for_supplier(invoice.supplier.party_id, seed)
    clean = rasterise(to_pdf(invoice, name), dpi)
    image, spec = degrade(clean, rng.uniform(*severity), rng)
    return image, name, spec


def render_to_file(invoice: Invoice, out_dir: Path, *, dpi: int, severity: tuple[float, float],
                   seed: int, quality: int = 92) -> RenderedDoc:
    image, template, spec = render_invoice(invoice, dpi=dpi, severity=severity, seed=seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{invoice.invoice_id}.jpg"
    image.save(path, format="JPEG", quality=quality)
    return RenderedDoc(invoice_id=invoice.invoice_id, business_id=invoice.business_id, split=invoice.split,
                       template=template, path=f"documents/{path.name}", width=image.width,
                       height=image.height, degradation=spec)
