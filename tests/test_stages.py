import random
from datetime import date, timedelta

import numpy as np
import pytest

from gstpipe.anomaly.cli import truth_record
from gstpipe.anomaly.detectors import RuleDetector, StatisticalDetector, build_detectors
from gstpipe.anomaly.evaluate import summarise
from gstpipe.anomaly.features import build as build_signals
from gstpipe.generator.anomalies import inject_anomalies
from gstpipe.generator.entities import make_business
from gstpipe.generator.invoices import make_invoices
from gstpipe.generator.transactions import make_transactions
from gstpipe.reconcile import features as recon_features
from gstpipe.reconcile.cli import _truth_view, build_candidates
from gstpipe.reconcile.evaluate import evaluate
from gstpipe.reconcile.matchers import ExactMatcher, FuzzyMatcher, LearnedMatcher, Prediction
from gstpipe.schema import TxnKind

START, END = date(2025, 4, 1), date(2026, 3, 31)


@pytest.fixture(scope="module")
def world():
    rng = random.Random(55)
    business = make_business(rng, "B000", "train", set())
    invoices = make_invoices(rng, business, START, END)
    txns, matches = make_transactions(rng, business, invoices, START, END)
    invoices, txns, matches, anomalies = inject_anomalies(rng, business, invoices, txns, matches,
                                                          END, rate=0.1)
    return invoices, txns, matches, anomalies


# ------------------------------------------------------------------ reconciliation

def test_candidate_rejects_payments_that_precede_the_invoice(world):
    invoices, txns, _, _ = world
    invoice, txn = invoices[0], txns[0]
    early = txn.model_copy(update={"txn_date": invoice.invoice_date - timedelta(days=40)})
    assert recon_features.build(early, invoice.invoice_id, invoice.supplier.name,
                                invoice.invoice_total, invoice.invoice_date, invoice.due_date,
                                invoice.invoice_number) is None


def test_vendor_similarity_survives_bank_truncation(world):
    invoices, txns, matches, _ = world
    by_id = {i.invoice_id: i for i in invoices}
    txn_by_id = {t.txn_id: t for t in txns}
    scores = []
    for match in matches[:40]:
        invoice, txn = by_id[match.invoice_id], txn_by_id[match.txn_id]
        candidate = recon_features.build(txn, invoice.invoice_id, invoice.supplier.name,
                                         invoice.invoice_total, invoice.invoice_date,
                                         invoice.due_date, invoice.invoice_number)
        if candidate:
            scores.append(candidate.features["vendor_partial"])
    assert scores and np.median(scores) > 0.6


def test_exact_matcher_is_precise_but_misses_tds_and_instalments(world):
    invoices, txns, matches, _ = world
    truth = {(m.txn_id, m.invoice_id) for m in matches}
    payable = {t.txn_id for t in txns if t.kind is TxnKind.invoice_payment}
    candidates = build_candidates([_truth_view(i) for i in invoices], txns)
    result = evaluate(ExactMatcher().predict(candidates), truth, payable)
    assert result.precision > 0.9
    assert result.recall < 0.8  # net-of-TDS and part payments never match to the paisa


def test_learned_matcher_beats_the_hand_weighted_one(world):
    invoices, txns, matches, _ = world
    truth = {(m.txn_id, m.invoice_id) for m in matches}
    payable = {t.txn_id for t in txns if t.kind is TxnKind.invoice_payment}
    candidates = build_candidates([_truth_view(i) for i in invoices], txns)
    labels = [int((c.txn_id, c.invoice_id) in truth) for c in candidates]

    fuzzy = evaluate(FuzzyMatcher().predict(candidates), truth, payable)
    learned = LearnedMatcher().fit(candidates, labels)
    fitted = evaluate(learned.predict(candidates), truth, payable)
    assert fitted.f1 > fuzzy.f1


def test_one_prediction_per_transaction_from_greedy_matchers(world):
    invoices, txns, _, _ = world
    candidates = build_candidates([_truth_view(i) for i in invoices], txns)
    predictions = FuzzyMatcher().predict(candidates)
    assert len({p.txn_id for p in predictions}) == len(predictions)


def test_review_flag_follows_the_confidence_threshold():
    assert Prediction("t", "i", 0.9).needs_review is False
    assert Prediction("t", "i", 0.2).needs_review is True


def test_recon_metrics_separate_false_matches_from_misses():
    truth = {("t1", "i1"), ("t2", "i2")}
    predictions = [Prediction("t1", "i1", 0.9), Prediction("t2", "i9", 0.8)]
    result = evaluate(predictions, truth, {"t1", "t2"})
    assert result.correct == 1 and result.false_matches == 1
    assert result.false_match_rate == 0.5 and result.unmatched_rate == 0.5


# ------------------------------------------------------------------ anomaly detection

def test_signals_come_back_in_the_order_they_were_given(world):
    """Regression: scores are aligned with the caller's list, not with chronological order."""
    invoices, _, _, _ = world
    records = [truth_record(i) for i in invoices]
    shuffled = list(records)
    random.Random(3).shuffle(shuffled)
    assert [s.invoice_id for s in build_signals(shuffled)] == [r.invoice_id for r in shuffled]


def test_structural_checks_fire_on_planted_anomalies(world):
    invoices, _, _, anomalies = world
    signals = {s.invoice_id: s for s in build_signals([truth_record(i) for i in invoices])}
    # A GST arithmetic error may break the sum, the CGST/SGST split, or the implied rate.
    structural = {"gst_arithmetic_error": {"gst_inconsistent", "tax_split_uneven", "implied_rate_invalid"},
                  "invalid_gstin": {"gstin_invalid"},
                  "due_before_invoice_date": {"due_before_invoice"},
                  "wrong_tax_type": {"tax_type_inconsistent"},
                  "duplicate_invoice": {"duplicate_number"}}
    checked = 0
    for anomaly in anomalies:
        expected = structural.get(anomaly.anomaly_type.value)
        if expected and anomaly.entity_type == "invoice":
            fired = set(signals[anomaly.entity_id].structural_failures)
            assert fired & expected, (anomaly, fired)
            checked += 1
    assert checked > 0


def test_clean_invoices_pass_every_structural_check(world):
    invoices, _, _, anomalies = world
    flagged = {a.entity_id for a in anomalies}
    for signal in build_signals([truth_record(i) for i in invoices]):
        if signal.invoice_id not in flagged:
            assert signal.structural_failures == []


def test_rule_detector_flags_exactly_the_structurally_broken(world):
    invoices, _, _, anomalies = world
    signals = build_signals([truth_record(i) for i in invoices])
    scores = RuleDetector().score(signals)
    expected = np.array([bool(s.structural_failures) for s in signals])
    assert ((scores >= 0.5) == expected).all()


def test_statistical_detector_scores_outliers_above_normal_invoices(world):
    invoices, _, _, anomalies = world
    signals = build_signals([truth_record(i) for i in invoices])
    detector = StatisticalDetector().fit(signals)
    scores = dict(zip((s.invoice_id for s in signals), detector.score(signals)))
    outliers = [a.entity_id for a in anomalies if a.anomaly_type.value == "amount_outlier"]
    if not outliers:
        pytest.skip("no amount outliers in this sample")
    flagged = {a.entity_id for a in anomalies}
    normal = [v for k, v in scores.items() if k not in flagged]
    assert np.mean([scores[o] for o in outliers]) > np.mean(normal)


def test_anomaly_summary_reports_imbalance_aware_metrics():
    labels = np.array([0, 0, 0, 1])
    scores = np.array([0.1, 0.2, 0.9, 0.95])
    result = summarise(labels, scores, threshold=0.5)
    assert result["recall"] == 1.0 and result["precision"] == 0.5
    assert result["false_positive_rate"] == pytest.approx(1 / 3, abs=1e-3)


def test_every_detector_returns_one_score_per_invoice(world):
    invoices, _, _, _ = world
    signals = build_signals([truth_record(i) for i in invoices])
    for name, detector in build_detectors().items():
        scores = detector.fit(signals).score(signals)
        assert len(scores) == len(signals), name
        assert np.isfinite(scores).all(), name
