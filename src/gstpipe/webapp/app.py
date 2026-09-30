"""The review application: what the pipeline read, what needs a human, and what was corrected.

Server-rendered so it runs with no build step. Every page has a JSON twin under /api, so a
separate front end can consume the same data without touching this layer.
"""

from __future__ import annotations

import os
from datetime import date
from types import SimpleNamespace
from uuid import uuid4
from decimal import Decimal
from pathlib import Path

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from gstpipe.extract.records import FIELDS, MONEY_FIELDS
from gstpipe.webapp import db

HERE = Path(__file__).parent
DATASET = Path(os.environ.get("GSTPIPE_DATASET", "data/synthetic/seed-2026"))
DB_PATH = Path(os.environ.get("GSTPIPE_DB", "data/review.db"))
MODEL_PATH = Path(os.environ.get("GSTPIPE_MODEL", "models/pipeline.joblib"))
UPLOAD_DIR = Path(os.environ.get("GSTPIPE_UPLOADS", "data/uploads"))
MAX_UPLOAD_BYTES = 20 * 1024 * 1024

engine = db.make_engine(DB_PATH)
db.create_all(engine)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="GST review", docs_url="/api/docs")
app.mount("/static", StaticFiles(directory=HERE / "static"), name="static")
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")
if (DATASET / "documents").exists():
    app.mount("/documents", StaticFiles(directory=DATASET / "documents"), name="documents")

_artifacts = None
_artifacts_error: str | None = None


def pipeline():
    """Load the trained pipeline once, and remember why it could not be loaded if it fails."""
    global _artifacts, _artifacts_error
    if _artifacts is None and _artifacts_error is None:
        try:
            from gstpipe.artifacts import load
            _artifacts = load(MODEL_PATH)
        except Exception as exc:  # surfaced on the upload page rather than crashing the app
            _artifacts_error = str(exc)
    return _artifacts

templates = Jinja2Templates(directory=HERE / "templates")
templates.env.filters["inr"] = lambda v: "-" if v is None else f"{Decimal(str(v)):,.2f}"
FIELD_LABELS = {
    "supplier_name": "Supplier", "supplier_gstin": "Supplier GSTIN",
    "recipient_name": "Billed to", "recipient_gstin": "Customer GSTIN",
    "invoice_number": "Invoice no.", "invoice_date": "Invoice date", "due_date": "Due date",
    "place_of_supply": "Place of supply", "taxable_total": "Taxable value",
    "cgst_total": "CGST", "sgst_total": "SGST", "igst_total": "IGST",
    "invoice_total": "Invoice total", "round_off": "Round off",
}
templates.env.filters["label"] = lambda f: FIELD_LABELS.get(f, str(f).replace("_", " ").capitalize())
templates.env.filters["short"] = lambda v, n=42: "-" if v is None else (
    str(v) if len(str(v)) <= n else str(v)[: n - 1] + "…")


def get_session():
    with Session(engine) as session:
        yield session


# ----------------------------------------------------------------- queries

def dashboard_data(session: Session, business_id: str | None = None) -> dict:
    invoices = select(db.Invoice)
    if business_id:
        invoices = invoices.where(db.Invoice.business_id == business_id)
    rows = session.scalars(invoices).all()

    by_category: dict[str, Decimal] = {}
    by_supplier: dict[str, Decimal] = {}
    for invoice in rows:
        amount = invoice.invoice_total or Decimal("0")
        leaf = (invoice.category or "Uncategorised").split(" > ")[-1]
        by_category[leaf] = by_category.get(leaf, Decimal("0")) + amount
        name = invoice.supplier_name or "Unknown supplier"
        by_supplier[name] = by_supplier.get(name, Decimal("0")) + amount

    payments = select(db.Payment)
    if business_id:
        payments = payments.where(db.Payment.business_id == business_id)
    payment_rows = session.scalars(payments).all()
    unmatched = [p for p in payment_rows if p.match_status == "unmatched" and p.kind == "invoice_payment"]

    open_flags = session.scalar(
        select(func.count(db.Flag.id)).join(db.Invoice).where(db.Flag.resolved.is_(False))
        .where(db.Invoice.business_id == business_id if business_id else True)) or 0

    return {
        "invoice_count": len(rows),
        "total_value": sum((i.invoice_total or Decimal("0")) for i in rows),
        "needs_review": sum(i.status == "needs_review" for i in rows),
        "approved": sum(i.status in ("approved", "auto_approved") for i in rows),
        "open_flags": open_flags,
        "unmatched_payments": len(unmatched),
        "unmatched_value": sum(p.amount for p in unmatched),
        "by_category": sorted(by_category.items(), key=lambda kv: kv[1], reverse=True)[:8],
        "by_supplier": sorted(by_supplier.items(), key=lambda kv: kv[1], reverse=True)[:8],
    }


def queue(session: Session, business_id: str | None = None) -> list[tuple[db.Invoice, list[db.Flag]]]:
    statement = select(db.Invoice).where(db.Invoice.status == "needs_review")
    if business_id:
        statement = statement.where(db.Invoice.business_id == business_id)
    invoices = session.scalars(statement).all()
    pairs = [(i, [f for f in i.flags if not f.resolved]) for i in invoices]
    pairs.sort(key=lambda pair: (-max((f.score for f in pair[1]), default=0.0),
                                 -(pair[0].invoice_total or Decimal("0"))))
    return pairs


# ----------------------------------------------------------------- pages

@app.get("/")
def home(request: Request, session: Session = Depends(get_session)):
    businesses = session.scalars(select(db.Business)).all()
    return templates.TemplateResponse(request, "dashboard.html", {
        "businesses": businesses, "stats": dashboard_data(session),
        "verification": verification_overview(session), "title": "Overview"})


@app.get("/businesses/{business_id}")
def business_page(business_id: str, request: Request, session: Session = Depends(get_session)):
    business = session.get(db.Business, business_id)
    if business is None:
        raise HTTPException(404, "no such business")
    return templates.TemplateResponse(request, "dashboard.html", {
        "businesses": session.scalars(select(db.Business)).all(),
        "business": business, "stats": dashboard_data(session, business_id),
        "verification": verification_overview(session, business_id), "title": business.name})


@app.get("/review")
def review(request: Request, state: str | None = None, business_id: str | None = None,
           session: Session = Depends(get_session)):
    from gstpipe.webapp.verification import STATE_LABEL, VERIFIED

    judged = [(invoice, assess_invoice(invoice))
              for invoice in _loaded_invoices(session, business_id)]
    if state in STATE_LABEL:
        rows = [pair for pair in judged if pair[1].state == state]
    else:
        rows = [pair for pair in judged if pair[1].state != VERIFIED]
    rows.sort(key=lambda pair: (pair[1].score, -(pair[0].invoice_total or 0)))

    counts = {key: sum(1 for _, v in judged if v.state == key) for key in STATE_LABEL}
    return templates.TemplateResponse(request, "review.html", {
        "rows": rows[:300], "total": len(rows), "state": state, "counts": counts,
        "labels": STATE_LABEL, "businesses": session.scalars(select(db.Business)).all(),
        "business": session.get(db.Business, business_id) if business_id else None,
        "title": "Review queue"})


@app.get("/invoices/{invoice_id}")
def invoice_page(invoice_id: str, request: Request, session: Session = Depends(get_session)):
    invoice = session.get(db.Invoice, invoice_id)
    if invoice is None:
        raise HTTPException(404, "no such invoice")
    fields = {f.field: f for f in invoice.fields}
    regions = {}
    for value in invoice.fields:
        if value.region:
            from gstpipe.extract.records import Region
            region = Region.model_validate_json(value.region)
            regions[value.field] = dict(region.box, text=region.text)
    events = session.scalars(select(db.AuditEvent).where(db.AuditEvent.invoice_id == invoice_id)
                             .order_by(db.AuditEvent.at.desc())).all()
    payments = session.scalars(select(db.Payment)
                               .where(db.Payment.matched_invoice_id == invoice_id)).all()
    return templates.TemplateResponse(request, "invoice.html", {
        "invoice": invoice, "fields": fields, "field_order": FIELDS,
        "money_fields": MONEY_FIELDS, "events": events, "payments": payments, "regions": regions,
        "verification": assess_invoice(invoice),
        "title": invoice.invoice_number or invoice.id})


@app.post("/invoices/{invoice_id}/save")
async def save_invoice(invoice_id: str, request: Request, session: Session = Depends(get_session)):
    invoice = session.get(db.Invoice, invoice_id)
    if invoice is None:
        raise HTTPException(404, "no such invoice")
    form = await request.form()
    existing = {f.field: f for f in invoice.fields}

    for field in FIELDS:
        submitted = (form.get(field) or "").strip() or None
        record = existing.get(field)
        if record is None:
            continue
        if submitted != record.approved_value:
            session.add(db.AuditEvent(invoice_id=invoice_id, field=field, action="correct",
                                      old_value=record.approved_value, new_value=submitted))
            record.approved_value = submitted
            record.corrected = submitted != record.model_value
            if hasattr(invoice, field):
                setattr(invoice, field, _coerce(field, submitted))
    session.commit()
    return RedirectResponse(f"/invoices/{invoice_id}", status_code=303)


@app.post("/invoices/{invoice_id}/approve")
def approve(invoice_id: str, session: Session = Depends(get_session)):
    invoice = session.get(db.Invoice, invoice_id)
    if invoice is None:
        raise HTTPException(404, "no such invoice")
    invoice.status = "approved"
    for flag in invoice.flags:
        flag.resolved = True
    session.add(db.AuditEvent(invoice_id=invoice_id, field=None, action="approve",
                              old_value=None, new_value="approved"))
    session.commit()
    return RedirectResponse("/review", status_code=303)


@app.get("/payments")
def payments_page(request: Request, session: Session = Depends(get_session)):
    rows = session.scalars(select(db.Payment).order_by(db.Payment.txn_date.desc()).limit(300)).all()
    summary = {status: session.scalar(select(func.count(db.Payment.id))
                                      .where(db.Payment.match_status == status)) or 0
               for status in ("matched", "needs_review", "unmatched", "not_payable")}
    return templates.TemplateResponse(request, "payments.html", {
        "payments": rows, "summary": summary, "title": "Payments"})


def _coerce(field: str, value: str | None):
    if value is None:
        return None
    if field in MONEY_FIELDS:
        try:
            return Decimal(value.replace(",", ""))
        except Exception:
            return None
    if field.endswith("_date"):
        try:
            return date.fromisoformat(value)
        except ValueError:
            return None
    return value


# ----------------------------------------------------------------- json api

@app.get("/api/stats")
def api_stats(business_id: str | None = None, session: Session = Depends(get_session)):
    stats = dashboard_data(session, business_id)
    return JSONResponse({k: (float(v) if isinstance(v, Decimal) else
                             [[a, float(b)] for a, b in v] if isinstance(v, list) else v)
                         for k, v in stats.items()})


@app.get("/api/queue")
def api_queue(session: Session = Depends(get_session)):
    return [{"invoice_id": invoice.id, "supplier": invoice.supplier_name,
             "total": float(invoice.invoice_total or 0),
             "flags": [{"kind": f.kind, "detail": f.detail, "score": f.score} for f in flags]}
            for invoice, flags in queue(session)]


@app.get("/api/invoices/{invoice_id}")
def api_invoice(invoice_id: str, session: Session = Depends(get_session)):
    invoice = session.get(db.Invoice, invoice_id)
    if invoice is None:
        raise HTTPException(404, "no such invoice")
    return {"invoice_id": invoice.id, "status": invoice.status,
            "fields": {f.field: {"model": f.model_value, "approved": f.approved_value,
                                 "corrected": f.corrected} for f in invoice.fields},
            "flags": [{"kind": f.kind, "detail": f.detail, "resolved": f.resolved}
                      for f in invoice.flags],
            "audit": [{"field": e.field, "action": e.action, "old": e.old_value,
                       "new": e.new_value, "at": e.at.isoformat()} for e in
                      session.scalars(select(db.AuditEvent)
                                      .where(db.AuditEvent.invoice_id == invoice_id)).all()]}


# ----------------------------------------------------------------- live processing

@app.get("/upload")
def upload_form(request: Request, session: Session = Depends(get_session)):
    ready = pipeline() is not None
    from gstpipe.extract.sources import tesseract_available
    return templates.TemplateResponse(request, "upload.html", {
        "businesses": session.scalars(select(db.Business)).all(),
        "model_ready": ready, "model_error": _artifacts_error,
        "model_meta": _artifacts.describe() if ready else None,
        "ocr_ready": tesseract_available(), "title": "Upload an invoice"})


@app.post("/upload")
async def upload(request: Request, file: UploadFile = File(...), business_id: str = Form(...),
                 session: Session = Depends(get_session)):
    from gstpipe.anomaly.cli import extracted_record
    from gstpipe.extract.records import ExtractedInvoice
    from gstpipe.webapp import processing

    artifacts = pipeline()
    if artifacts is None:
        raise HTTPException(503, f"no trained pipeline available: {_artifacts_error}")
    if session.get(db.Business, business_id) is None:
        raise HTTPException(400, "choose a business to file this invoice under")

    data = await file.read()
    if not data:
        raise HTTPException(400, "that file was empty")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "file is larger than 20 MB")

    invoice_id = f"UP-{uuid4().hex[:10].upper()}"
    history = [
        extracted_record(
            SimpleNamespace(invoice_id=row.id, business_id=row.business_id),
            ExtractedInvoice(document_id=row.id, supplier_name=row.supplier_name,
                             supplier_gstin=row.supplier_gstin, invoice_number=row.invoice_number,
                             invoice_date=row.invoice_date, due_date=row.due_date,
                             place_of_supply=row.place_of_supply, taxable_total=row.taxable_total,
                             cgst_total=row.cgst_total, sgst_total=row.sgst_total,
                             igst_total=row.igst_total, invoice_total=row.invoice_total))
        for row in session.scalars(select(db.Invoice)
                                   .where(db.Invoice.business_id == business_id).limit(600)).all()]

    try:
        result = processing.process(data, file.filename or "", invoice_id, business_id,
                                    artifacts, history)
    except Exception as exc:
        raise HTTPException(422, f"could not read that document: {exc}") from exc

    preview = UPLOAD_DIR / f"{invoice_id}.jpg"
    result.image.convert("RGB").save(preview, format="JPEG", quality=88)

    extracted = result.extracted
    session.add(db.Invoice(
        id=invoice_id, business_id=business_id,
        supplier_name=extracted.supplier_name, supplier_gstin=extracted.supplier_gstin,
        invoice_number=extracted.invoice_number, invoice_date=extracted.invoice_date,
        due_date=extracted.due_date, place_of_supply=extracted.place_of_supply,
        taxable_total=extracted.taxable_total, cgst_total=extracted.cgst_total,
        sgst_total=extracted.sgst_total, igst_total=extracted.igst_total,
        invoice_total=extracted.invoice_total, category=result.category,
        document_path=f"uploads/{preview.name}",
        status="needs_review" if result.needs_review else "auto_approved",
        extraction_confidence=result.ocr_confidence))
    for field in FIELDS:
        value = getattr(extracted, field, None)
        text = None if value is None else str(value)
        region = extracted.provenance.get(field)
        session.add(db.FieldValue(invoice_id=invoice_id, field=field, model_value=text,
                                  approved_value=text,
                                  region=region.model_dump_json() if region else None))
    for flag in result.flags:
        session.add(db.Flag(invoice_id=invoice_id, kind=flag.kind, detail=flag.detail,
                            score=flag.score))
    session.add(db.AuditEvent(invoice_id=invoice_id, field=None, action="upload", old_value=None,
                              new_value=file.filename, actor="upload"))
    session.commit()
    return RedirectResponse(f"/invoices/{invoice_id}", status_code=303)


@app.post("/api/process")
async def api_process(file: UploadFile = File(...)):
    """Run a document through the pipeline and return the result without storing anything."""
    from gstpipe.webapp import processing

    artifacts = pipeline()
    if artifacts is None:
        raise HTTPException(503, f"no trained pipeline available: {_artifacts_error}")
    data = await file.read()
    if not data:
        raise HTTPException(400, "that file was empty")
    try:
        result = processing.process(data, file.filename or "", "preview", "preview", artifacts)
    except Exception as exc:
        raise HTTPException(422, f"could not read that document: {exc}") from exc
    return {"fields": result.extracted.model_dump(mode="json"), "category": result.category,
            "ocr_confidence": round(result.ocr_confidence, 4),
            "flags": [{"kind": f.kind, "detail": f.detail, "score": round(f.score, 4)}
                      for f in result.flags]}


@app.get("/api/health")
def health():
    from gstpipe.extract.sources import tesseract_available
    artifacts = pipeline()
    return {"status": "ok" if artifacts and tesseract_available() else "degraded",
            "model": artifacts.meta if artifacts else None, "model_error": _artifacts_error,
            "ocr": tesseract_available()}


@app.get("/sample.pdf")
def sample_invoice():
    """A clean invoice to try the upload with, rendered on demand from the test split."""
    import random

    from fastapi.responses import Response

    from gstpipe.render.pipeline import TEMPLATE_NAMES, to_pdf
    from gstpipe.schema import Invoice

    source = DATASET / "ground_truth" / "invoices.jsonl"
    if not source.exists():
        raise HTTPException(404, "no dataset available to draw a sample from")
    lines = source.read_text("utf-8").splitlines()
    invoice = Invoice.model_validate_json(random.choice(lines))
    pdf = to_pdf(invoice, random.choice(TEMPLATE_NAMES))
    return Response(pdf, media_type="application/pdf", headers={
        "Content-Disposition": f'attachment; filename="sample-{invoice.invoice_number}.pdf"'})


# ----------------------------------------------------------------- analytics and source files

@app.get("/analytics")
def analytics_page(request: Request, business_id: str | None = None,
                   session: Session = Depends(get_session)):
    from gstpipe.webapp.analytics import build

    businesses = session.scalars(select(db.Business)).all()
    business = session.get(db.Business, business_id) if business_id else None
    return templates.TemplateResponse(request, "analytics.html", {
        "businesses": businesses, "business": business,
        "data": build(session, business_id),
        "title": f"Analytics · {business.name}" if business else "Analytics"})


@app.get("/documents")
def documents_page(request: Request, business_id: str | None = None, page: int = 1,
                   session: Session = Depends(get_session)):
    """Every source file the figures came from, in one place."""
    per_page = 48
    statement = select(db.Invoice).where(db.Invoice.document_path.is_not(None))
    if business_id:
        statement = statement.where(db.Invoice.business_id == business_id)
    rows = session.scalars(statement.order_by(db.Invoice.invoice_date.desc())).all()

    total = len(rows)
    page = max(1, page)
    window = rows[(page - 1) * per_page: page * per_page]
    uploaded = sum(1 for r in rows if (r.document_path or "").startswith("uploads/"))
    return templates.TemplateResponse(request, "documents.html", {
        "businesses": session.scalars(select(db.Business)).all(),
        "business": session.get(db.Business, business_id) if business_id else None,
        "documents": window, "total": total, "page": page,
        "pages": max(1, -(-total // per_page)), "uploaded": uploaded,
        "title": "Source files"})


@app.get("/api/analytics")
def api_analytics(business_id: str | None = None, session: Session = Depends(get_session)):
    from gstpipe.webapp.analytics import build

    data = build(session, business_id)
    return JSONResponse({
        "as_of": data["as_of"].isoformat(),
        "invoice_count": data["invoice_count"],
        "total_spend": float(data["total_spend"]),
        "input_credit": float(data["input_credit"]),
        "outstanding": {"count": data["outstanding_count"],
                        "value": float(data["outstanding_value"])},
        "ageing": [{**row, "value": float(row["value"])} for row in data["ageing"]],
        "monthly_spend": [{"month": label, "value": float(value)}
                          for label, value in zip(data["spend"].labels, data["spend"].values)],
        "top_suppliers": [{**row, "value": float(row["value"])} for row in data["suppliers"]],
        "average_days_to_pay": data["average_days_to_pay"],
    })


# ----------------------------------------------------------------- verification

def assess_invoice(invoice: db.Invoice):
    """Judge one invoice, using which of its fields could be located on the page."""
    from gstpipe.webapp.verification import assess

    traced = {f.field for f in invoice.fields if f.region}
    return assess(invoice, signed_off=invoice.status == "approved", traced=traced)


def _loaded_invoices(session: Session, business_id: str | None = None) -> list[db.Invoice]:
    from sqlalchemy.orm import selectinload

    statement = select(db.Invoice).options(selectinload(db.Invoice.fields),
                                           selectinload(db.Invoice.flags))
    if business_id:
        statement = statement.where(db.Invoice.business_id == business_id)
    return list(session.scalars(statement).all())


def verification_overview(session: Session, business_id: str | None = None) -> dict:
    from gstpipe.webapp.verification import NEEDS_CHECK, UNREADABLE, VERIFIED, summarise

    invoices = _loaded_invoices(session, business_id)
    judged = [(invoice, assess_invoice(invoice)) for invoice in invoices]
    states = summarise([(v, i.invoice_total) for i, v in judged])

    worst: dict[str, int] = {}
    for _, verification in judged:
        for check in verification.failed:
            worst[check.name] = worst.get(check.name, 0) + 1

    signed = sum(1 for i, _ in judged if i.status == "approved")
    return {
        "states": states,
        "total": len(judged),
        "verified_value": states[VERIFIED]["value"],
        "unverified_value": states[NEEDS_CHECK]["value"] + states[UNREADABLE]["value"],
        "signed_off": signed,
        "blockers": sorted(worst.items(), key=lambda kv: kv[1], reverse=True),
        "queue": sorted(
            ((i, v) for i, v in judged if v.state != VERIFIED),
            key=lambda pair: (pair[1].state != NEEDS_CHECK, -(pair[0].invoice_total or 0)))[:10],
    }
