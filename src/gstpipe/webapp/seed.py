"""Load a generated dataset into the review database.

What lands in the app is the pipeline's own output -- OCR extraction, detector flags, matcher
results -- not ground truth. The point of the screen is to show a CA what the system believes and
let them correct it, so seeding with the true answers would demonstrate nothing.
"""

from __future__ import annotations

import json
from pathlib import Path

from sqlalchemy.orm import Session

from gstpipe.anomaly.cli import extracted_record, truth_record
from gstpipe.anomaly.detectors import build_detectors
from gstpipe.anomaly.features import build as build_signals
from gstpipe.extract.records import FIELDS, ExtractedInvoice
from gstpipe.reconcile.cli import (_extracted_view, _truth_view, build_candidates, load_matches,
                                   load_transactions)
from gstpipe.reconcile.matchers import build_matchers
from gstpipe.schema import Invoice, TxnKind
from gstpipe.webapp import db

FLAG_THRESHOLD = 0.5
CRITICAL_FIELDS = ("supplier_gstin", "invoice_number", "invoice_date", "invoice_total")


def _load_invoices(dataset: Path) -> list[Invoice]:
    path = dataset / "ground_truth" / "invoices.jsonl"
    return [Invoice.model_validate_json(line) for line in path.read_text("utf-8").splitlines()]


def _load_extractions(dataset: Path) -> tuple[dict[str, ExtractedInvoice], str]:
    for source in ("ocr", "text"):
        path = dataset / "extractions" / f"{source}.jsonl"
        if path.exists():
            records = {}
            for line in path.read_text("utf-8").splitlines():
                record = ExtractedInvoice.model_validate(json.loads(line))
                records[record.document_id] = record
            return records, source
    return {}, "none"


def _as_text(value) -> str | None:
    return None if value is None else str(value)


def seed(dataset: Path, session: Session, businesses: int | None = 3, progress=print) -> dict:
    invoices = _load_invoices(dataset)
    extractions, source = _load_extractions(dataset)
    progress(f"extractions: {len(extractions)} from {source}")

    chosen = sorted({i.business_id for i in invoices})[:businesses] if businesses else None
    if chosen is not None:
        invoices = [i for i in invoices if i.business_id in chosen]
    wanted = {i.business_id for i in invoices}

    documents = {}
    docs_csv = dataset / "ground_truth" / "documents.csv"
    if docs_csv.exists():
        import pandas as pd
        frame = pd.read_csv(docs_csv)
        documents = dict(zip(frame.invoice_id, frame.file))

    # Businesses
    meta = {}
    for line in (dataset / "ground_truth" / "businesses.jsonl").read_text("utf-8").splitlines():
        record = json.loads(line)
        meta[record["party"]["party_id"]] = record
    for business_id in sorted(wanted):
        record = meta[business_id]
        session.add(db.Business(id=business_id, name=record["party"]["name"],
                                sector=record["sector"], state_code=record["party"]["state_code"],
                                gstin=record["party"]["gstin"]))

    # Detectors and matcher, fitted on clean training data as they would be in production
    all_invoices = _load_invoices(dataset)
    train = [i for i in all_invoices if i.split == "train"]
    detectors = build_detectors()
    train_signals = build_signals([truth_record(i) for i in train])
    for detector in detectors.values():
        detector.fit(train_signals)

    txns = [t for t in load_transactions(dataset) if t.business_id in wanted]
    truth_pairs = load_matches(dataset)
    matchers = build_matchers()
    train_txns = [t for t in load_transactions(dataset) if t.split == "train"]
    train_candidates = build_candidates([_truth_view(i) for i in train], train_txns)
    train_labels = [int((c.txn_id, c.invoice_id) in truth_pairs) for c in train_candidates]
    matcher = matchers["m3_forest"]
    matcher.fit(train_candidates, train_labels)
    progress("fitted detectors and matcher on training data")

    records = [extracted_record(i, extractions.get(i.invoice_id)) for i in invoices]
    signals = build_signals(records)
    rule_scores = detectors["a1_rules"].score(signals)
    model_scores = detectors["a3_isolation_forest"].score(signals)

    for invoice, signal, rule_score, model_score in zip(invoices, signals, rule_scores, model_scores):
        extracted = extractions.get(invoice.invoice_id)
        values = {f: getattr(extracted, f, None) for f in FIELDS} if extracted else dict.fromkeys(FIELDS)
        confidence = float((extracted.confidence or {}).get("ocr_mean", 0.0)) if extracted else 0.0

        flags = []
        for failure in signal.structural_failures:
            flags.append(db.Flag(invoice_id=invoice.invoice_id, kind=failure,
                                 detail=f"consistency check failed: {failure.replace('_', ' ')}",
                                 score=1.0))
        if model_score >= FLAG_THRESHOLD:
            flags.append(db.Flag(invoice_id=invoice.invoice_id, kind="model_outlier",
                                 detail="isolation forest scored this unusual for the supplier",
                                 score=float(model_score)))
        missing = [f for f in CRITICAL_FIELDS if values.get(f) is None]
        if missing:
            flags.append(db.Flag(invoice_id=invoice.invoice_id, kind="missing_fields",
                                 detail="could not read: " + ", ".join(missing), score=0.9))

        session.add(db.Invoice(
            id=invoice.invoice_id, business_id=invoice.business_id,
            supplier_name=_as_text(values.get("supplier_name")),
            supplier_gstin=_as_text(values.get("supplier_gstin")),
            invoice_number=_as_text(values.get("invoice_number")),
            invoice_date=values.get("invoice_date"), due_date=values.get("due_date"),
            place_of_supply=_as_text(values.get("place_of_supply")),
            taxable_total=values.get("taxable_total"), cgst_total=values.get("cgst_total"),
            sgst_total=values.get("sgst_total"), igst_total=values.get("igst_total"),
            invoice_total=values.get("invoice_total"), category=invoice.category,
            document_path=documents.get(invoice.invoice_id),
            status="needs_review" if flags else "auto_approved",
            extraction_confidence=confidence))
        regions = (extracted.provenance if extracted else {}) or {}
        for field in FIELDS:
            text = _as_text(values.get(field))
            region = regions.get(field)
            session.add(db.FieldValue(invoice_id=invoice.invoice_id, field=field,
                                      model_value=text, approved_value=text,
                                      region=region.model_dump_json() if region else None))
        for flag in flags:
            session.add(flag)

    # Payments, matched with the extracted view of each invoice
    views = [_extracted_view(i, extractions.get(i.invoice_id)) for i in invoices]
    candidates = build_candidates(views, txns)
    predictions = {p.txn_id: p for p in matcher.predict(candidates)}
    for txn in txns:
        prediction = predictions.get(txn.txn_id)
        payable = txn.kind is TxnKind.invoice_payment
        status = "unmatched"
        if prediction:
            status = "matched" if not prediction.needs_review else "needs_review"
        elif not payable:
            status = "not_payable"
        session.add(db.Payment(id=txn.txn_id, business_id=txn.business_id, txn_date=txn.txn_date,
                               amount=txn.amount, narration=txn.narration, kind=txn.kind.value,
                               matched_invoice_id=prediction.invoice_id if prediction else None,
                               match_score=prediction.score if prediction else None,
                               match_status=status))

    session.commit()
    counts = {"businesses": len(wanted), "invoices": len(invoices), "payments": len(txns),
              "flagged": sum(1 for s in signals if s.structural_failures)}
    return counts


def seed_database(dataset: Path, db_path: Path, businesses: int | None = 3, progress=print) -> dict:
    engine = db.make_engine(db_path)
    db.reset(engine)
    with Session(engine) as session:
        counts = seed(dataset, session, businesses, progress)
    return counts
