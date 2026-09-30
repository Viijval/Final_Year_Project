"""Render a generated dataset to document images: ``gstpipe-render --dataset data/synthetic/seed-2026``."""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

from gstpipe.render.pipeline import render_to_file
from gstpipe.schema import Invoice


def _worker(args) -> dict:
    invoice_json, out_dir, dpi, severity, seed = args
    return render_to_file(Invoice.model_validate_json(invoice_json), Path(out_dir),
                          dpi=dpi, severity=severity, seed=seed).as_row()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Render invoices to scanned-looking document images.")
    parser.add_argument("--dataset", type=Path, required=True, help="dataset directory holding ground_truth/")
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--severity-min", type=float, default=0.05)
    parser.add_argument("--severity-max", type=float, default=0.85)
    parser.add_argument("--limit", type=int, help="render only the first N invoices (for a quick look)")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--workers", type=int, default=0, help="0 = one per CPU")
    args = parser.parse_args(argv)

    source = args.dataset / "ground_truth" / "invoices.jsonl"
    if not source.exists():
        print(f"no invoices at {source}; generate a dataset first", file=sys.stderr)
        return 1

    lines = source.read_text(encoding="utf-8").splitlines()
    if args.limit:
        lines = lines[:args.limit]
    out_dir = args.dataset / "documents"
    severity = (args.severity_min, args.severity_max)
    tasks = [(line, str(out_dir), args.dpi, severity, args.seed) for line in lines]

    rows = []
    with ProcessPoolExecutor(max_workers=args.workers or None) as pool:
        for i, row in enumerate(pool.map(_worker, tasks, chunksize=16), start=1):
            rows.append(row)
            if i % 250 == 0 or i == len(tasks):
                print(f"  rendered {i}/{len(tasks)}", flush=True)

    index = pd.DataFrame(rows).sort_values("invoice_id")
    index.to_csv(args.dataset / "ground_truth" / "documents.csv", index=False)
    print(f"wrote {out_dir} ({len(index)} documents)")
    print("  templates: " + ", ".join(f"{k}={v}" for k, v in index.template.value_counts().items()))
    print(f"  severity: {index.severity.min():.2f}-{index.severity.max():.2f}, "
          f"median blur {index.blur_radius.median():.2f}px, median JPEG quality {int(index.jpeg_quality.median())}")
    print(f"  grayscale {index.grayscale.mean():.0%}, shadowed {index.shadow.mean():.0%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
