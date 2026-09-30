"""Run Stage 1 over a dataset and score it: ``gstpipe-extract --dataset ... --source ocr``.

Two sources: ``text`` reads the PDF's own text layer (a perfect reader, the upper bound) and
``ocr`` reads the degraded document image (the realistic case). The gap between them is the
extraction error that the rest of the pipeline has to live with.
"""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd
from PIL import Image

from gstpipe.extract.evaluate import Outcome, compare, score
from gstpipe.extract.records import FIELDS
from gstpipe.extract.rules import extract
from gstpipe.extract.sources import from_image, from_pdf, tesseract_available
from gstpipe.render.pipeline import to_pdf
from gstpipe.schema import Invoice

BUCKETS = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.01)]


def _run_one(args) -> tuple[str, dict, dict]:
    invoice_json, dataset, template, source, file = args
    invoice = Invoice.model_validate_json(invoice_json)
    if source == "text":
        page = from_pdf(to_pdf(invoice, template))
    else:
        with Image.open(Path(dataset) / file) as img:
            page = from_image(img)
    extracted = extract(page, invoice.invoice_id)
    outcomes = compare(invoice, extracted)
    return invoice.invoice_id, {f: o.value for f, o in outcomes.items()}, extracted.model_dump(mode="json")


def _report(frame: pd.DataFrame, by: str) -> pd.DataFrame:
    rows = []
    for key, group in frame.groupby(by, observed=True):
        flat = group[list(FIELDS)].to_numpy().ravel()
        total = len(flat)
        correct = int((flat == "correct").sum())
        wrong = int((flat == "wrong").sum())
        rows.append({by: key, "documents": len(group), "accuracy": round(correct / total, 4),
                     "precision": round(correct / (correct + wrong), 4) if correct + wrong else 0.0,
                     "missing_rate": round(int((flat == "missing").sum()) / total, 4)})
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Extract fields from rendered invoices and score them.")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--source", choices=("ocr", "text"), default="ocr")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--workers", type=int, default=0)
    args = parser.parse_args(argv)

    gt = args.dataset / "ground_truth"
    docs_csv = gt / "documents.csv"
    if not docs_csv.exists():
        print(f"no documents.csv in {gt}; render the dataset first", file=sys.stderr)
        return 1
    if args.source == "ocr" and not tesseract_available():
        print("tesseract not found; install it or use --source text", file=sys.stderr)
        return 1

    docs = pd.read_csv(docs_csv)
    invoices = {}
    for line in (gt / "invoices.jsonl").read_text(encoding="utf-8").splitlines():
        invoices[line.split('"invoice_id":"', 1)[1].split('"', 1)[0]] = line
    docs = docs[docs.invoice_id.isin(invoices)]
    if args.limit:
        docs = docs.head(args.limit)

    tasks = [(invoices[r.invoice_id], str(args.dataset), r.template, args.source, r.file)
             for r in docs.itertuples()]
    print(f"extracting {len(tasks)} documents via {args.source}...")

    outcomes, extractions = {}, []
    with ProcessPoolExecutor(max_workers=args.workers or None) as pool:
        for i, (invoice_id, row, payload) in enumerate(pool.map(_run_one, tasks, chunksize=8), start=1):
            outcomes[invoice_id] = row
            extractions.append(payload)
            if i % 250 == 0 or i == len(tasks):
                print(f"  {i}/{len(tasks)}", flush=True)

    frame = docs.set_index("invoice_id").join(pd.DataFrame(outcomes).T)
    frame["severity_bucket"] = pd.cut(frame.severity, [b[0] for b in BUCKETS] + [BUCKETS[-1][1]],
                                      right=False, precision=2)

    overall, per_field = score([{f: Outcome(v) for f, v in row.items()} for row in outcomes.values()])

    out_dir = args.dataset / "reports"
    out_dir.mkdir(exist_ok=True)
    (args.dataset / "extractions").mkdir(exist_ok=True)
    pd.DataFrame(extractions).to_json(args.dataset / "extractions" / f"{args.source}.jsonl",
                                      orient="records", lines=True)
    field_table = pd.DataFrame({f: s.as_row() for f, s in per_field.items()}).T.sort_values("accuracy")
    field_table.to_csv(out_dir / f"extraction_{args.source}_by_field.csv")
    _report(frame, "template").to_csv(out_dir / f"extraction_{args.source}_by_template.csv", index=False)
    _report(frame, "severity_bucket").to_csv(out_dir / f"extraction_{args.source}_by_severity.csv", index=False)

    print(f"\noverall field accuracy {overall.accuracy:.3f}  precision {overall.precision:.3f}  "
          f"F1 {overall.f1:.3f}  (missing {overall.missing}/{overall.total})")
    print("\nworst fields:")
    print(field_table.head(6).to_string())
    print("\nby template:")
    print(_report(frame, "template").to_string(index=False))
    print("\nby severity:")
    print(_report(frame, "severity_bucket").to_string(index=False))
    print(f"\nwrote {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
