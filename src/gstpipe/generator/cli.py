"""Command-line entry point: ``gstpipe-generate --businesses 30 --seed 2026``."""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

from gstpipe.generator.dataset import DatasetConfig, build_dataset, manifest, write_dataset


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate a synthetic GST invoice dataset with ground truth.")
    parser.add_argument("--out", type=Path, help="output directory (default: data/synthetic/seed-<seed>)")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--businesses", type=int, default=30)
    parser.add_argument("--start", type=date.fromisoformat, default=date(2025, 4, 1))
    parser.add_argument("--end", type=date.fromisoformat, default=date(2026, 3, 31))
    parser.add_argument("--anomaly-rate", type=float, default=0.04)
    parser.add_argument("--force", action="store_true", help="write into a non-empty output directory")
    args = parser.parse_args(argv)

    out = args.out or Path("data") / "synthetic" / f"seed-{args.seed}"
    if out.exists() and any(out.iterdir()) and not args.force:
        print(f"{out} already contains files; pass --force to overwrite", file=sys.stderr)
        return 1

    config = DatasetConfig(seed=args.seed, n_businesses=args.businesses, start=args.start, end=args.end,
                           anomaly_rate=args.anomaly_rate)
    ds = build_dataset(config)
    write_dataset(ds, out)

    info = manifest(ds)
    print(f"wrote {out}")
    for split, counts in info["counts_by_split"].items():
        print(f"  {split:<5} " + "  ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    print("  anomalies: " + ", ".join(f"{k}={v}" for k, v in sorted(info["anomaly_types"].items())))
    print("  matches:   " + ", ".join(f"{k}={v}" for k, v in sorted(info["match_types"].items())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
