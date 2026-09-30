"""Run a single uploaded document through the whole pipeline.

This is the live path: bytes in, structured invoice plus review flags out. It uses the artifacts
trained offline, so nothing is fitted while a request is in flight.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from decimal import Decimal

import pymupdf
from PIL import Image

from gstpipe.anomaly.features import InvoiceRecord, InvoiceSignals
from gstpipe.anomaly.features import build as build_signals
from gstpipe.artifacts import Artifacts
from gstpipe.categorise.features import from_extracted
from gstpipe.extract.records import ExtractedInvoice
from gstpipe.extract.rules import extract
from gstpipe.extract.sources import from_image, tesseract_available
from gstpipe.render.pipeline import rasterise

RENDER_DPI = 200
MAX_PIXELS = 4000
CRITICAL_FIELDS = ("supplier_gstin", "invoice_number", "invoice_date", "invoice_total")
MODEL_FLAG_THRESHOLD = 0.5


@dataclass
class Flag:
    kind: str
    detail: str
    score: float


@dataclass
class ProcessedDocument:
    invoice_id: str
    extracted: ExtractedInvoice
    category: str
    signals: InvoiceSignals
    ocr_confidence: float
    image: Image.Image
    flags: list[Flag] = field(default_factory=list)

    @property
    def needs_review(self) -> bool:
        return bool(self.flags)


def to_image(data: bytes, filename: str = "") -> Image.Image:
    """Accept a PDF or an image; a PDF is rasterised at the resolution OCR expects."""
    if data[:5] == b"%PDF-" or filename.lower().endswith(".pdf"):
        with pymupdf.open(stream=data, filetype="pdf") as doc:
            if doc.page_count == 0:
                raise ValueError("that PDF has no pages")
        return rasterise(data, RENDER_DPI)
    image = Image.open(io.BytesIO(data))
    image.load()
    if max(image.size) > MAX_PIXELS:
        scale = MAX_PIXELS / max(image.size)
        image = image.resize((int(image.width * scale), int(image.height * scale)))
    return image.convert("RGB")


def process(data: bytes, filename: str, invoice_id: str, business_id: str,
            artifacts: Artifacts, history: list[InvoiceRecord] | None = None) -> ProcessedDocument:
    if not tesseract_available():
        raise RuntimeError("tesseract is not installed, so uploaded documents cannot be read")

    image = to_image(data, filename)
    page = from_image(image)
    extracted = extract(page, invoice_id)
    confidence = float((extracted.confidence or {}).get("ocr_mean", 0.0))

    category = artifacts.categoriser.predict([from_extracted(extracted)])[0]

    record = InvoiceRecord(
        invoice_id=invoice_id, business_id=business_id, supplier_name=extracted.supplier_name,
        supplier_gstin=extracted.supplier_gstin, invoice_number=extracted.invoice_number,
        invoice_date=extracted.invoice_date, due_date=extracted.due_date,
        place_of_supply=extracted.place_of_supply, taxable_total=extracted.taxable_total,
        cgst_total=extracted.cgst_total, sgst_total=extracted.sgst_total,
        igst_total=extracted.igst_total, invoice_total=extracted.invoice_total,
        round_off=extracted.round_off, n_lines=len(extracted.line_items))

    # History matters: duplicate numbers and unusual amounts are only visible against past invoices.
    batch = list(history or []) + [record]
    signals = {s.invoice_id: s for s in build_signals(batch)}[invoice_id]

    flags = [Flag(name, f"consistency check failed: {name.replace('_', ' ')}", 1.0)
             for name in signals.structural_failures]
    score = float(artifacts.detector.score([signals])[0])
    if score >= MODEL_FLAG_THRESHOLD:
        flags.append(Flag("model_outlier", "scored unusual against this supplier's history", score))
    missing = [f for f in CRITICAL_FIELDS if getattr(extracted, f) is None]
    if missing:
        flags.append(Flag("missing_fields", "could not read: " + ", ".join(missing), 0.9))
    if confidence and confidence < 0.6:
        flags.append(Flag("low_confidence", f"OCR confidence {confidence:.0%}", 0.7))

    return ProcessedDocument(invoice_id=invoice_id, extracted=extracted, category=category,
                             signals=signals, ocr_confidence=confidence, image=image, flags=flags)


def as_decimal(value) -> Decimal | None:
    return None if value is None else Decimal(str(value))
