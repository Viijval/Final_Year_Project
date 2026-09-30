"""E5: how far an extraction error travels.

Every stage is trained once on clean data, the way it would be in practice. Then the test split is
run through the whole chain at injected upstream error rates, and each stage is measured at each
rate. Injection is random, so every rate is repeated across seeds and reported as mean and spread.

Two quantities come out of it:

    degradation   D(s, e) = M(s, 0) - M(s, e)
    amplification A(s, e) = D(s, e) / e

A(s, e) above 1 means the stage loses more accuracy than the error it was handed -- it amplifies.
Below 1 means it absorbs, most plausibly through signals that survived the damage.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import pandas as pd

from gstpipe.anomaly.cli import extracted_record, truth_record
from gstpipe.anomaly.detectors import build_detectors
from gstpipe.anomaly.evaluate import summarise as anomaly_summary
from gstpipe.anomaly.features import build as build_signals
from gstpipe.categorise.evaluate import summarise as category_summary
from gstpipe.categorise.features import from_extracted, from_invoice
from gstpipe.categorise.models import build_models
from gstpipe.extract.evaluate import Outcome, compare
from gstpipe.extract.records import ExtractedInvoice
from gstpipe.inject.corrupt import InjectionReport, corrupt, from_invoice as perfect_extraction
from gstpipe.reconcile.cli import _extracted_view, _truth_view, build_candidates
from gstpipe.reconcile.evaluate import evaluate as reconcile_evaluate
from gstpipe.reconcile.matchers import build_matchers
from gstpipe.schema import Invoice, Transaction, TxnKind

RATES = (0.0, 0.05, 0.10, 0.20, 0.30)
SEEDS = (1, 2, 3, 4, 5)


@dataclass
class Fitted:
    categorisers: dict
    matchers: dict
    detectors: dict


def train(train_invoices: list[Invoice], train_txns: list[Transaction],
          truth_pairs: set[tuple[str, str]], seed: int = 0) -> Fitted:
    """Fit every stage once, on clean training data."""
    categorisers = build_models(seed)
    inputs = [from_invoice(i) for i in train_invoices]
    labels = [i.category for i in train_invoices]
    for model in categorisers.values():
        model.fit(inputs, labels)

    candidates = build_candidates([_truth_view(i) for i in train_invoices], train_txns)
    pair_labels = [int((c.txn_id, c.invoice_id) in truth_pairs) for c in candidates]
    matchers = build_matchers(seed)
    for matcher in matchers.values():
        matcher.fit(candidates, pair_labels)

    detectors = build_detectors(seed)
    signals = build_signals([truth_record(i) for i in train_invoices])
    for detector in detectors.values():
        detector.fit(signals)

    return Fitted(categorisers, matchers, detectors)


def _field_accuracy(invoices: list[Invoice], views: dict[str, ExtractedInvoice]) -> float:
    total = correct = 0
    for invoice in invoices:
        outcomes = compare(invoice, views[invoice.invoice_id])
        total += len(outcomes)
        correct += sum(o is Outcome.correct for o in outcomes.values())
    return correct / total if total else 0.0


def run_once(fitted: Fitted, invoices: list[Invoice], txns: list[Transaction],
             truth_pairs: set[tuple[str, str]], flagged: set[str], rate: float,
             seed: int) -> list[dict]:
    """One pass of the whole chain at a single injected error rate."""
    rng = random.Random(f"{seed}:{rate}")
    report = InjectionReport(rate)
    views = {i.invoice_id: corrupt(perfect_extraction(i), rate, rng, report) for i in invoices}

    rows = [{"stage": "1_extraction", "method": "injected", "metric": "field_accuracy",
             "value": _field_accuracy(invoices, views)},
            {"stage": "1_extraction", "method": "injected", "metric": "realised_rate",
             "value": report.realised_rate}]

    category_inputs = [from_extracted(views[i.invoice_id]) for i in invoices]
    category_truth = [i.category for i in invoices]
    for name, model in fitted.categorisers.items():
        scores = category_summary(category_truth, model.predict(category_inputs))
        rows.append({"stage": "2_categorisation", "method": name, "metric": "accuracy",
                     "value": scores["accuracy"]})
        rows.append({"stage": "2_categorisation", "method": name, "metric": "macro_f1",
                     "value": scores["macro_f1"]})

    recon_views = [_extracted_view(i, views[i.invoice_id]) for i in invoices]
    candidates = build_candidates(recon_views, txns)
    ids = {i.invoice_id for i in invoices}
    truth_subset = {(t, i) for t, i in truth_pairs if i in ids}
    payable = {t.txn_id for t in txns if t.kind is TxnKind.invoice_payment}
    for name, matcher in fitted.matchers.items():
        result = reconcile_evaluate(matcher.predict(candidates), truth_subset, payable)
        rows.append({"stage": "3_reconciliation", "method": name, "metric": "f1", "value": result.f1})
        rows.append({"stage": "3_reconciliation", "method": name, "metric": "false_match_rate",
                     "value": result.false_match_rate})

    signals = build_signals([extracted_record(i, views[i.invoice_id]) for i in invoices])
    labels = np.array([int(i.invoice_id in flagged) for i in invoices])
    for name, detector in fitted.detectors.items():
        scores = anomaly_summary(labels, detector.score(signals))
        for metric in ("f1", "precision", "recall", "false_positive_rate", "flagged_rate",
                       "average_precision"):
            rows.append({"stage": "4_anomaly", "method": name, "metric": metric,
                         "value": scores[metric]})

    for row in rows:
        row |= {"rate": rate, "seed": seed}
    return rows


def sweep(fitted: Fitted, invoices, txns, truth_pairs, flagged,
          rates=RATES, seeds=SEEDS, progress=print) -> pd.DataFrame:
    rows: list[dict] = []
    for rate in rates:
        for seed in seeds:
            rows += run_once(fitted, invoices, txns, truth_pairs, flagged, rate, seed)
        progress(f"  rate {rate:.0%} done ({len(seeds)} seeds)")
    return pd.DataFrame(rows)


def summarise(frame: pd.DataFrame) -> pd.DataFrame:
    """Mean and spread per condition, with degradation and amplification against the clean run."""
    grouped = (frame.groupby(["stage", "method", "metric", "rate"], observed=True)["value"]
               .agg(["mean", "std", "count"]).reset_index())
    baseline = (grouped[grouped.rate == 0.0]
                .set_index(["stage", "method", "metric"])["mean"].rename("baseline"))
    merged = grouped.join(baseline, on=["stage", "method", "metric"])
    merged["degradation"] = merged.baseline - merged["mean"]
    merged["amplification"] = np.where(merged.rate > 0, merged.degradation / merged.rate, np.nan)
    return merged.sort_values(["stage", "method", "metric", "rate"])


def amplification_table(summary: pd.DataFrame, headline: dict[str, str] | None = None) -> pd.DataFrame:
    """One row per stage and method: the headline metric's amplification at each rate."""
    headline = headline or {"1_extraction": "field_accuracy", "2_categorisation": "accuracy",
                            "3_reconciliation": "f1", "4_anomaly": "f1"}
    mask = summary.apply(lambda r: headline.get(r.stage) == r.metric, axis=1)
    view = summary[mask & (summary.rate > 0)]
    return view.pivot_table(index=["stage", "method"], columns="rate", values="amplification")
