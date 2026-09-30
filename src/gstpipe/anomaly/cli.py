"""Train and score Stage 4: ``python -m gstpipe.anomaly.cli --dataset ...``."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from gstpipe.anomaly.detectors import build_detectors
from gstpipe.anomaly.evaluate import summarise
from gstpipe.anomaly.features import InvoiceRecord, build
from gstpipe.extract.records import ExtractedInvoice
from gstpipe.schema import Invoice


def truth_record(invoice: Invoice) -> InvoiceRecord:
    return InvoiceRecord(
        invoice_id=invoice.invoice_id, business_id=invoice.business_id,
        supplier_name=invoice.supplier.name, supplier_gstin=invoice.supplier.gstin,
        invoice_number=invoice.invoice_number, invoice_date=invoice.invoice_date,
        due_date=invoice.due_date, place_of_supply=invoice.place_of_supply,
        taxable_total=invoice.taxable_total, cgst_total=invoice.cgst_total,
        sgst_total=invoice.sgst_total, igst_total=invoice.igst_total,
        invoice_total=invoice.invoice_total, round_off=invoice.round_off,
        n_lines=len(invoice.line_items))


def extracted_record(invoice: Invoice, extracted: ExtractedInvoice | None) -> InvoiceRecord:
    if extracted is None:
        # Nothing was read from this document: every field is unknown, not zero.
        return InvoiceRecord(
            invoice_id=invoice.invoice_id, business_id=invoice.business_id, supplier_name=None,
            supplier_gstin=None, invoice_number=None, invoice_date=None, due_date=None,
            place_of_supply=None, taxable_total=None, cgst_total=None, sgst_total=None,
            igst_total=None, invoice_total=None, round_off=None, n_lines=0)
    return InvoiceRecord(
        invoice_id=invoice.invoice_id, business_id=invoice.business_id,
        supplier_name=extracted.supplier_name, supplier_gstin=extracted.supplier_gstin,
        invoice_number=extracted.invoice_number, invoice_date=extracted.invoice_date,
        due_date=extracted.due_date, place_of_supply=extracted.place_of_supply,
        taxable_total=extracted.taxable_total, cgst_total=extracted.cgst_total,
        sgst_total=extracted.sgst_total, igst_total=extracted.igst_total,
        invoice_total=extracted.invoice_total, round_off=extracted.round_off,
        n_lines=len(extracted.line_items))


def load_extractions(dataset: Path, source: str) -> dict[str, ExtractedInvoice]:
    path = dataset / "extractions" / f"{source}.jsonl"
    if not path.exists():
        return {}
    return {r.document_id: r for r in
            (ExtractedInvoice.model_validate(json.loads(line))
             for line in path.read_text("utf-8").splitlines())}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Flag invoices for review and score the detectors.")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args(argv)

    invoices = [Invoice.model_validate_json(line) for line in
                (args.dataset / "ground_truth" / "invoices.jsonl").read_text("utf-8").splitlines()]
    anomalies = pd.read_csv(args.dataset / "ground_truth" / "anomalies.csv")
    flagged = set(anomalies[anomalies.entity_type == "invoice"].entity_id)

    by_split = defaultdict(list)
    for invoice in invoices:
        by_split[invoice.split].append(invoice)

    train_signals = build([truth_record(i) for i in by_split["train"]])
    test_invoices = by_split["test"]
    labels = np.array([int(i.invoice_id in flagged) for i in test_invoices])
    print(f"train {len(train_signals)} invoices, test {len(test_invoices)} "
          f"({labels.sum()} anomalies, {labels.mean():.2%})")

    conditions = {"truth": build([truth_record(i) for i in test_invoices])}
    for source in ("text", "ocr"):
        extracted = load_extractions(args.dataset, source)
        if extracted:
            conditions[f"extracted_{source}"] = build(
                [extracted_record(i, extracted.get(i.invoice_id)) for i in test_invoices])
        else:
            print(f"  (no {source} extractions yet; skipping that condition)")

    detectors = build_detectors(args.seed)
    for name, detector in detectors.items():
        detector.fit(train_signals)
        print(f"  fitted {name}")

    rows = []
    for condition, signals in conditions.items():
        for name, detector in detectors.items():
            scores = detector.score(signals)
            rows.append({"detector": name, "condition": condition,
                         **summarise(labels, scores, args.threshold)})

    table = pd.DataFrame(rows)
    out = args.dataset / "reports"
    out.mkdir(exist_ok=True)
    table.to_csv(out / "anomaly.csv", index=False)
    print("\nF1 by condition:")
    print(table.pivot(index="detector", columns="condition", values="f1").to_string())
    print("\nfull metrics:")
    print(table.to_string(index=False))
    print(f"\nwrote {out / 'anomaly.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
