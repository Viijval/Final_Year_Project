"""Run the propagation sweep: ``python -m gstpipe.inject.cli --dataset ...``."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import pandas as pd

from gstpipe.inject import plots
from gstpipe.inject.experiment import (RATES, SEEDS, amplification_table, summarise, sweep, train)
from gstpipe.reconcile.cli import load_matches, load_transactions
from gstpipe.schema import Invoice


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Measure how extraction errors propagate downstream.")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--seeds", type=int, default=len(SEEDS))
    parser.add_argument("--seed", type=int, default=0, help="seed for model fitting")
    args = parser.parse_args(argv)

    gt = args.dataset / "ground_truth"
    invoices = [Invoice.model_validate_json(line)
                for line in (gt / "invoices.jsonl").read_text("utf-8").splitlines()]
    txns = load_transactions(args.dataset)
    truth_pairs = load_matches(args.dataset)
    anomalies = pd.read_csv(gt / "anomalies.csv")
    flagged = set(anomalies[anomalies.entity_type == "invoice"].entity_id)

    by_split = defaultdict(list)
    for invoice in invoices:
        by_split[invoice.split].append(invoice)
    txns_by_split = defaultdict(list)
    for txn in txns:
        txns_by_split[txn.split].append(txn)

    print(f"fitting stages on {len(by_split['train'])} training invoices...")
    fitted = train(by_split["train"], txns_by_split["train"], truth_pairs, args.seed)

    seeds = tuple(SEEDS[:args.seeds])
    print(f"sweeping rates {[f'{r:.0%}' for r in RATES]} across {len(seeds)} seeds "
          f"on {len(by_split['test'])} test invoices")
    raw = sweep(fitted, by_split["test"], txns_by_split["test"], truth_pairs, flagged, seeds=seeds)

    summary = summarise(raw)
    amplification = amplification_table(summary)

    out = args.dataset / "reports"
    out.mkdir(exist_ok=True)
    raw.to_csv(out / "propagation_raw.csv", index=False)
    summary.to_csv(out / "propagation_summary.csv", index=False)
    amplification.to_csv(out / "propagation_amplification.csv")
    figures = plots.write_all(summary, out / "figures")

    headline = {"1_extraction": "field_accuracy", "2_categorisation": "accuracy",
                "3_reconciliation": "f1", "4_anomaly": "f1"}
    mask = summary.apply(lambda r: headline.get(r.stage) == r.metric, axis=1)
    curves = summary[mask].pivot_table(index=["stage", "method"], columns="rate", values="mean")
    print("\nHeadline metric by injected error rate:")
    print(curves.round(3).to_string())
    print("\nAmplification factor A(s, e):")
    print(amplification.round(2).to_string())
    print("\nfigures: " + ", ".join(f.name for f in figures))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
