"""Train and score Stage 3: ``python -m gstpipe.reconcile.cli --dataset ...``.

Bank transactions are structured data in this pipeline, so they are never degraded. Extraction
noise reaches reconciliation only through the invoice side: a misread amount, supplier name, date
or invoice number. That is precisely the channel the propagation study measures.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from pathlib import Path

import pandas as pd

from gstpipe.extract.records import ExtractedInvoice
from gstpipe.reconcile import features
from gstpipe.reconcile.evaluate import evaluate
from gstpipe.reconcile.matchers import build_matchers
from gstpipe.schema import Invoice, Transaction, TxnKind


@dataclass(frozen=True)
class InvoiceView:
    """What the matcher believes about an invoice: either ground truth or Stage 1 output."""

    invoice_id: str
    business_id: str
    supplier_name: str | None
    invoice_total: Decimal | None
    invoice_date: date | None
    due_date: date | None
    invoice_number: str | None


def _truth_view(invoice: Invoice) -> InvoiceView:
    return InvoiceView(invoice.invoice_id, invoice.business_id, invoice.supplier.name,
                       invoice.invoice_total, invoice.invoice_date, invoice.due_date,
                       invoice.invoice_number)


def _extracted_view(invoice: Invoice, extracted: ExtractedInvoice | None) -> InvoiceView:
    if extracted is None:
        return InvoiceView(invoice.invoice_id, invoice.business_id, None, None, None, None, None)
    return InvoiceView(invoice.invoice_id, invoice.business_id, extracted.supplier_name,
                       extracted.invoice_total, extracted.invoice_date, extracted.due_date,
                       extracted.invoice_number)


def load_transactions(dataset: Path) -> list[Transaction]:
    frame = pd.read_csv(dataset / "ground_truth" / "transactions.csv", dtype=str).fillna("")
    out = []
    for row in frame.to_dict("records"):
        out.append(Transaction(txn_id=row["txn_id"], business_id=row["business_id"],
                               split=row["split"], txn_date=date.fromisoformat(row["txn_date"]),
                               amount=Decimal(row["amount"]), mode=row["mode"],
                               narration=row["narration"], kind=TxnKind(row["kind"]),
                               counterparty_id=row["counterparty_id"] or None))
    return out


def load_matches(dataset: Path) -> set[tuple[str, str]]:
    frame = pd.read_csv(dataset / "ground_truth" / "matches.csv", dtype=str)
    return set(zip(frame.txn_id, frame.invoice_id))


def load_extractions(dataset: Path, source: str) -> dict[str, ExtractedInvoice]:
    path = dataset / "extractions" / f"{source}.jsonl"
    if not path.exists():
        return {}
    records = {}
    for line in path.read_text("utf-8").splitlines():
        record = ExtractedInvoice.model_validate(json.loads(line))
        records[record.document_id] = record
    return records


def build_candidates(views: list[InvoiceView], txns: list[Transaction]) -> list[features.Candidate]:
    by_business: dict[str, list[InvoiceView]] = defaultdict(list)
    for view in views:
        by_business[view.business_id].append(view)

    candidates = []
    for txn in txns:
        for view in by_business.get(txn.business_id, ()):
            candidate = features.build(txn, view.invoice_id, view.supplier_name, view.invoice_total,
                                       view.invoice_date, view.due_date, view.invoice_number)
            if candidate is not None:
                candidates.append(candidate)
    return candidates


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Match bank payments to invoices and score it.")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    invoices = [Invoice.model_validate_json(line) for line in
                (args.dataset / "ground_truth" / "invoices.jsonl").read_text("utf-8").splitlines()]
    txns = load_transactions(args.dataset)
    truth_pairs = load_matches(args.dataset)

    by_split = defaultdict(list)
    for invoice in invoices:
        by_split[invoice.split].append(invoice)
    txns_by_split = defaultdict(list)
    for txn in txns:
        txns_by_split[txn.split].append(txn)

    train_candidates = build_candidates([_truth_view(i) for i in by_split["train"]],
                                        txns_by_split["train"])
    train_labels = [int((c.txn_id, c.invoice_id) in truth_pairs) for c in train_candidates]
    positives = sum(train_labels)
    print(f"train candidates {len(train_candidates)} ({positives} true settlements, "
          f"{positives / len(train_candidates):.2%} positive)")

    test_invoices = by_split["test"]
    test_txns = txns_by_split["test"]
    test_ids = {i.invoice_id for i in test_invoices}
    test_truth = {(t, i) for t, i in truth_pairs if i in test_ids}
    payable = {t.txn_id for t in test_txns if t.kind is TxnKind.invoice_payment}

    conditions = {"truth": [_truth_view(i) for i in test_invoices]}
    for source in ("text", "ocr"):
        extracted = load_extractions(args.dataset, source)
        if extracted:
            conditions[f"extracted_{source}"] = [
                _extracted_view(i, extracted.get(i.invoice_id)) for i in test_invoices]
        else:
            print(f"  (no {source} extractions yet; skipping that condition)")

    matchers = build_matchers(args.seed)
    for name, matcher in matchers.items():
        matcher.fit(train_candidates, train_labels)
        print(f"  fitted {name}")

    rows = []
    for condition, views in conditions.items():
        candidates = build_candidates(views, test_txns)
        for name, matcher in matchers.items():
            result = evaluate(matcher.predict(candidates), test_truth, payable)
            rows.append({"matcher": name, "condition": condition, **result.as_row()})

    table = pd.DataFrame(rows)
    out = args.dataset / "reports"
    out.mkdir(exist_ok=True)
    table.to_csv(out / "reconciliation.csv", index=False)
    print("\nF1 by condition:")
    print(table.pivot(index="matcher", columns="condition", values="f1").to_string())
    print("\nfull metrics:")
    print(table.to_string(index=False))
    print(f"\nwrote {out / 'reconciliation.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
