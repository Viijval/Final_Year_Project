"""Assemble a full synthetic dataset and write it with its ground-truth register and manifest."""

from __future__ import annotations

import json
import platform
import random
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import date
from importlib.metadata import version
from pathlib import Path

import pandas as pd

from gstpipe import __version__
from gstpipe.generator.anomalies import inject_anomalies
from gstpipe.generator.entities import Business, make_business
from gstpipe.generator.invoices import make_invoices
from gstpipe.generator.transactions import make_transactions
from gstpipe.schema import Anomaly, Invoice, Match, Split, Transaction


@dataclass(frozen=True)
class DatasetConfig:
    seed: int = 2026
    n_businesses: int = 30
    split_ratios: tuple[float, float, float] = (0.7, 0.15, 0.15)
    start: date = date(2025, 4, 1)
    end: date = date(2026, 3, 31)
    anomaly_rate: float = 0.04
    vendors_per_category: tuple[int, int] = (1, 3)


@dataclass
class Dataset:
    config: DatasetConfig
    businesses: list[Business] = field(default_factory=list)
    invoices: list[Invoice] = field(default_factory=list)
    transactions: list[Transaction] = field(default_factory=list)
    matches: list[Match] = field(default_factory=list)
    anomalies: list[Anomaly] = field(default_factory=list)


def assign_splits(n: int, ratios: tuple[float, float, float], rng: random.Random) -> list[Split]:
    if n < 3:
        raise ValueError("need at least 3 businesses so every split is non-empty")
    n_val = max(1, round(n * ratios[1]))
    n_test = max(1, round(n * ratios[2]))
    splits: list[Split] = ["train"] * (n - n_val - n_test) + ["val"] * n_val + ["test"] * n_test
    rng.shuffle(splits)
    return splits


def build_dataset(config: DatasetConfig) -> Dataset:
    rng = random.Random(config.seed)
    ds = Dataset(config=config)
    used_names: set[str] = set()
    for index, split in enumerate(assign_splits(config.n_businesses, config.split_ratios, rng)):
        business = make_business(rng, f"B{index:03d}", split, used_names, config.vendors_per_category)
        invoices = make_invoices(rng, business, config.start, config.end)
        txns, matches = make_transactions(rng, business, invoices, config.start, config.end)
        invoices, txns, matches, anomalies = inject_anomalies(
            rng, business, invoices, txns, matches, config.end, config.anomaly_rate)
        ds.businesses.append(business)
        ds.invoices += invoices
        ds.transactions += txns
        ds.matches += matches
        ds.anomalies += anomalies
    return ds


def _rows(models) -> pd.DataFrame:
    return pd.DataFrame([m.model_dump(mode="json") for m in models])


def manifest(ds: Dataset) -> dict:
    split_of = {b.party.party_id: b.split for b in ds.businesses}
    counts = {s: Counter() for s in ("train", "val", "test")}
    for inv in ds.invoices:
        counts[inv.split]["invoices"] += 1
    for t in ds.transactions:
        counts[t.split]["transactions"] += 1
    for a in ds.anomalies:
        counts[split_of[a.entity_id.split("-")[0]]]["anomalies"] += 1
    for b in ds.businesses:
        counts[b.split]["businesses"] += 1
    return {
        "generator_version": __version__,
        "config": {k: (v.isoformat() if isinstance(v, date) else v) for k, v in asdict(ds.config).items()},
        "counts_by_split": {s: dict(c) for s, c in counts.items()},
        "anomaly_types": dict(Counter(a.anomaly_type.value for a in ds.anomalies)),
        "match_types": dict(Counter(m.match_type.value for m in ds.matches)),
        "transaction_kinds": dict(Counter(t.kind.value for t in ds.transactions)),
        "businesses": [{"business_id": b.party.party_id, "split": b.split, "sector": b.sector,
                        "state_code": b.party.state_code, "vendors": len(b.vendors)} for b in ds.businesses],
        "environment": {"python": platform.python_version(),
                        **{pkg: version(pkg) for pkg in ("pydantic", "numpy", "pandas")}},
    }


def write_dataset(ds: Dataset, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    gt = out_dir / "ground_truth"
    gt.mkdir(exist_ok=True)

    with (gt / "invoices.jsonl").open("w", encoding="utf-8") as fh:
        fh.writelines(inv.model_dump_json() + "\n" for inv in ds.invoices)
    with (gt / "businesses.jsonl").open("w", encoding="utf-8") as fh:
        for b in ds.businesses:
            record = {"split": b.split, "sector": b.sector, "party": b.party.model_dump(mode="json"),
                      "vendors": [{"category": v.category, "party": v.party.model_dump(mode="json")}
                                  for v in b.vendors]}
            fh.write(json.dumps(record) + "\n")
    _rows(ds.transactions).to_csv(gt / "transactions.csv", index=False)
    _rows(ds.matches).to_csv(gt / "matches.csv", index=False)
    _rows(ds.anomalies).to_csv(gt / "anomalies.csv", index=False)

    (out_dir / "manifest.json").write_text(json.dumps(manifest(ds), indent=2), encoding="utf-8")
    return out_dir
