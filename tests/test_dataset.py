import json
from datetime import date

import pandas as pd
import pytest

from gstpipe.generator.cli import main
from gstpipe.generator.dataset import DatasetConfig, build_dataset, write_dataset
from gstpipe.schema import Invoice

CONFIG = DatasetConfig(seed=3, n_businesses=8, start=date(2025, 7, 1), end=date(2026, 2, 28))


@pytest.fixture(scope="module")
def written(tmp_path_factory):
    ds = build_dataset(CONFIG)
    return ds, write_dataset(ds, tmp_path_factory.mktemp("ds"))


def test_every_split_is_populated_and_vendors_never_cross(written):
    ds, _ = written
    assert {b.split for b in ds.businesses} == {"train", "val", "test"}
    owner = {}
    for b in ds.businesses:
        for v in b.vendors:
            assert owner.setdefault(v.party.name, b.split) == b.split


def test_files_round_trip_without_loss(written):
    ds, out = written
    gt = out / "ground_truth"
    reloaded = [Invoice.model_validate_json(line) for line in (gt / "invoices.jsonl").read_text("utf-8").splitlines()]
    assert reloaded == ds.invoices
    assert len(pd.read_csv(gt / "transactions.csv")) == len(ds.transactions)
    assert len(pd.read_csv(gt / "matches.csv")) == len(ds.matches)
    assert len(pd.read_csv(gt / "anomalies.csv")) == len(ds.anomalies)


def test_manifest_counts_agree_with_records(written):
    ds, out = written
    info = json.loads((out / "manifest.json").read_text("utf-8"))
    assert sum(c.get("invoices", 0) for c in info["counts_by_split"].values()) == len(ds.invoices)
    assert sum(info["anomaly_types"].values()) == len(ds.anomalies)
    assert info["config"]["seed"] == CONFIG.seed and "pandas" in info["environment"]


def test_build_is_deterministic():
    a, b = build_dataset(CONFIG), build_dataset(CONFIG)
    assert [i.model_dump_json() for i in a.invoices] == [i.model_dump_json() for i in b.invoices]
    assert [t.model_dump_json() for t in a.transactions] == [t.model_dump_json() for t in b.transactions]


def test_cli_refuses_to_overwrite_without_force(tmp_path, capsys):
    args = ["--out", str(tmp_path), "--businesses", "3", "--start", "2025-10-01", "--end", "2025-12-31"]
    (tmp_path / "keep.txt").write_text("x")
    assert main(args) == 1
    assert main(args + ["--force"]) == 0
    assert (tmp_path / "manifest.json").exists()
