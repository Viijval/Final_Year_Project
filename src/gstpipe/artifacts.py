"""Train the pipeline once and persist it, so the application loads a model instead of fitting one.

Everything here is fitted on the training split only. The application never trains: it loads these
artifacts at start-up and applies them to whatever document arrives.
"""

from __future__ import annotations

import json
import platform
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import joblib
from sklearn.ensemble import IsolationForest

from gstpipe import __version__
from gstpipe.anomaly.cli import truth_record
from gstpipe.anomaly.detectors import ModelDetector, RuleDetector, StatisticalDetector
from gstpipe.anomaly.features import build as build_signals
from gstpipe.categorise.features import from_invoice
from gstpipe.categorise.models import Categoriser, build_models
from gstpipe.reconcile.cli import _truth_view, build_candidates, load_matches, load_transactions
from gstpipe.reconcile.matchers import LearnedMatcher, build_matchers
from gstpipe.schema import Invoice

DEFAULT_PATH = Path("models/pipeline.joblib")
CATEGORISER = "hybrid_lr"
MATCHER = "m3_forest"
DETECTOR = "a3_isolation_forest"


@dataclass
class Artifacts:
    categoriser: Categoriser
    matcher: LearnedMatcher
    detector: ModelDetector
    statistical: StatisticalDetector
    rules: RuleDetector
    meta: dict

    def describe(self) -> str:
        return (f"gstpipe {self.meta.get('version')} · trained on "
                f"{self.meta.get('train_invoices')} invoices · {self.meta.get('trained_at', '')[:10]}")


def train(dataset: Path, progress=print) -> Artifacts:
    invoices = [Invoice.model_validate_json(line) for line in
                (dataset / "ground_truth" / "invoices.jsonl").read_text("utf-8").splitlines()]
    train_invoices = [i for i in invoices if i.split == "train"]
    if not train_invoices:
        raise ValueError(f"no training invoices in {dataset}")

    progress(f"training on {len(train_invoices)} invoices")
    categoriser = build_models()[CATEGORISER]
    categoriser.fit([from_invoice(i) for i in train_invoices], [i.category for i in train_invoices])
    progress(f"  categoriser {CATEGORISER} fitted")

    txns = [t for t in load_transactions(dataset) if t.split == "train"]
    truth_pairs = load_matches(dataset)
    candidates = build_candidates([_truth_view(i) for i in train_invoices], txns)
    labels = [int((c.txn_id, c.invoice_id) in truth_pairs) for c in candidates]
    matcher = build_matchers()[MATCHER]
    matcher.fit(candidates, labels)
    progress(f"  matcher {MATCHER} fitted on {len(candidates)} candidate pairs")

    signals = build_signals([truth_record(i) for i in train_invoices])
    detector = ModelDetector(DETECTOR, IsolationForest(n_estimators=300, contamination=0.05,
                                                       random_state=0, n_jobs=-1)).fit(signals)
    statistical = StatisticalDetector().fit(signals)
    progress(f"  detectors fitted on {len(signals)} signals")

    meta = {"version": __version__, "trained_at": datetime.now(timezone.utc).isoformat(),
            "dataset": str(dataset), "train_invoices": len(train_invoices),
            "categoriser": CATEGORISER, "matcher": MATCHER, "detector": DETECTOR,
            "python": platform.python_version()}
    return Artifacts(categoriser, matcher, detector, statistical, RuleDetector(), meta)


def save(artifacts: Artifacts, path: Path = DEFAULT_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifacts, path, compress=3)
    path.with_suffix(".json").write_text(json.dumps(artifacts.meta, indent=2), encoding="utf-8")
    return path


def load(path: Path = DEFAULT_PATH) -> Artifacts:
    if not Path(path).exists():
        raise FileNotFoundError(
            f"no trained pipeline at {path}. Run: python -m gstpipe.artifacts --dataset <dataset>")
    return joblib.load(path)


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Train and save the pipeline used by the application.")
    parser.add_argument("--dataset", type=Path, default=Path("data/synthetic/seed-2026"))
    parser.add_argument("--out", type=Path, default=DEFAULT_PATH)
    args = parser.parse_args(argv)

    artifacts = train(args.dataset)
    path = save(artifacts, args.out)
    size = path.stat().st_size / 1_000_000
    print(f"wrote {path} ({size:.1f} MB)")
    print(artifacts.describe())
    return 0


if __name__ == "__main__":
    # Re-import under the real module name: running with -m makes this __main__, and anything
    # pickled from here would then be unloadable from any other process.
    from gstpipe.artifacts import main as entry

    raise SystemExit(entry())
