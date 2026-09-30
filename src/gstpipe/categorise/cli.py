"""Train and score Stage 2: ``python -m gstpipe.categorise.cli --dataset ...``.

Models are trained once on clean training data, the way they would be in practice, then scored on
the test split under each input condition: ground truth, perfect-reader extraction, and OCR.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from gstpipe.categorise.evaluate import summarise
from gstpipe.categorise.features import CategoryInput, from_extracted, from_invoice
from gstpipe.categorise.models import build_models
from gstpipe.categorise.rules import predict as rule_predict
from gstpipe.extract.records import ExtractedInvoice
from gstpipe.schema import Invoice


def load_invoices(dataset: Path) -> list[Invoice]:
    path = dataset / "ground_truth" / "invoices.jsonl"
    return [Invoice.model_validate_json(line) for line in path.read_text("utf-8").splitlines()]


def load_extracted(dataset: Path, source: str) -> dict[str, CategoryInput]:
    path = dataset / "extractions" / f"{source}.jsonl"
    if not path.exists():
        return {}
    out = {}
    for line in path.read_text("utf-8").splitlines():
        record = ExtractedInvoice.model_validate(json.loads(line))
        out[record.document_id] = from_extracted(record)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train and evaluate invoice categorisation.")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    invoices = load_invoices(args.dataset)
    if not invoices:
        print("no invoices found; generate a dataset first", file=sys.stderr)
        return 1

    train = [i for i in invoices if i.split == "train"]
    test = [i for i in invoices if i.split == "test"]
    train_inputs = [from_invoice(i) for i in train]
    train_labels = [i.category for i in train]
    test_labels = [i.category for i in test]
    print(f"train {len(train)} invoices, test {len(test)}, {len(set(train_labels))} categories")

    conditions: dict[str, list[CategoryInput]] = {"truth": [from_invoice(i) for i in test]}
    for source in ("text", "ocr"):
        extracted = load_extracted(args.dataset, source)
        if extracted:
            conditions[f"extracted_{source}"] = [
                extracted.get(i.invoice_id, CategoryInput(invoice_id=i.invoice_id)) for i in test]
        else:
            print(f"  (no {source} extractions yet; skipping that condition)")

    models = build_models(args.seed)
    for name, model in models.items():
        model.fit(train_inputs, train_labels)
        print(f"  trained {name}")

    rows = []
    for condition, inputs in conditions.items():
        rows.append({"model": "rules_c1", "condition": condition,
                     **summarise(test_labels, [rule_predict(x) for x in inputs])})
        for name, model in models.items():
            rows.append({"model": name, "condition": condition,
                         **summarise(test_labels, model.predict(inputs))})

    table = pd.DataFrame(rows)
    out = args.dataset / "reports"
    out.mkdir(exist_ok=True)
    table.to_csv(out / "categorisation.csv", index=False)

    print("\n" + table.pivot(index="model", columns="condition", values="accuracy").to_string())
    print("\nfull metrics:")
    print(table.to_string(index=False))
    print(f"\nwrote {out / 'categorisation.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
