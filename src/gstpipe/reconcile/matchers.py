"""Four matching tiers, from an exact-amount lookup to a globally optimal assignment."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from gstpipe.reconcile.features import Candidate

REVIEW_THRESHOLD = 0.55


@dataclass(frozen=True)
class Prediction:
    txn_id: str
    invoice_id: str
    score: float

    @property
    def needs_review(self) -> bool:
        return self.score < REVIEW_THRESHOLD


def _best_per_txn(scored: list[tuple[Candidate, float]], threshold: float) -> list[Prediction]:
    best: dict[str, tuple[Candidate, float]] = {}
    for candidate, score in scored:
        if score < threshold:
            continue
        if candidate.txn_id not in best or score > best[candidate.txn_id][1]:
            best[candidate.txn_id] = (candidate, score)
    return [Prediction(c.txn_id, c.invoice_id, float(s)) for c, s in best.values()]


class ExactMatcher:
    """M1: the amount must agree to the paisa, and the payment must follow the invoice."""

    name = "m1_exact"

    def fit(self, candidates, labels):
        return self

    def predict(self, candidates: list[Candidate]) -> list[Prediction]:
        scored = [(c, 1.0) for c in candidates
                  if c.features["amount_exact"] and 0 <= c.features["days_after_invoice"] <= 0.4]
        return _best_per_txn(scored, 0.5)


class FuzzyMatcher:
    """M2: hand-weighted similarity, the way a rules-based reconciliation tool is tuned."""

    name = "m2_fuzzy"
    WEIGHTS = {"amount_exact": 0.34, "tds_plausible": 0.16, "amount_close": 0.06,
               "vendor_similarity": 0.24, "vendor_partial": 0.06, "invoice_number_hit": 0.22,
               "within_terms": 0.02}

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold

    def fit(self, candidates, labels):
        return self

    def score(self, candidate: Candidate) -> float:
        raw = sum(weight * candidate.features[name] for name, weight in self.WEIGHTS.items())
        penalty = 0.12 * min(1.0, candidate.features["log_amount_gap"] / 6.0)
        return max(0.0, min(1.0, raw - penalty))

    def predict(self, candidates: list[Candidate]) -> list[Prediction]:
        return _best_per_txn([(c, self.score(c)) for c in candidates], self.threshold)


class LearnedMatcher:
    """M3: the same signals, with the weights learned from labelled settlements."""

    def __init__(self, name: str = "m3_learned", estimator=None, threshold: float = 0.5):
        self.name = name
        # `or` would call __len__ on an unfitted ensemble, which raises; test for None explicitly.
        self.estimator = estimator if estimator is not None else LogisticRegression(
            max_iter=2000, class_weight="balanced")
        self.threshold = threshold

    def fit(self, candidates: list[Candidate], labels: list[int]) -> LearnedMatcher:
        self.estimator.fit(np.array([c.vector for c in candidates]), np.asarray(labels))
        return self

    def scores(self, candidates: list[Candidate]) -> np.ndarray:
        if not candidates:
            return np.zeros(0)
        return self.estimator.predict_proba(np.array([c.vector for c in candidates]))[:, 1]

    def predict(self, candidates: list[Candidate]) -> list[Prediction]:
        return _best_per_txn(list(zip(candidates, self.scores(candidates))), self.threshold)


class AssignmentMatcher:
    """M4: global one-to-one assignment instead of each payment greedily taking its best invoice.

    Greedy matching commits locally and cannot revise; solving the whole business at once resolves
    contention between payments competing for the same invoice. The cost is that genuinely
    one-to-many settlements (a consolidated transfer, an instalment) break the one-to-one assumption.
    """

    def __init__(self, scorer: LearnedMatcher, threshold: float = 0.5, name: str = "m4_assignment"):
        self.scorer = scorer
        self.threshold = threshold
        self.name = name

    def fit(self, candidates, labels):
        return self

    def predict(self, candidates: list[Candidate]) -> list[Prediction]:
        if not candidates:
            return []
        scores = self.scorer.scores(candidates)
        txns = sorted({c.txn_id for c in candidates})
        invoices = sorted({c.invoice_id for c in candidates})
        row = {t: i for i, t in enumerate(txns)}
        col = {v: i for i, v in enumerate(invoices)}
        matrix = np.zeros((len(txns), len(invoices)))
        for candidate, score in zip(candidates, scores):
            matrix[row[candidate.txn_id], col[candidate.invoice_id]] = score
        rows, cols = linear_sum_assignment(matrix, maximize=True)
        return [Prediction(txns[r], invoices[c], float(matrix[r, c]))
                for r, c in zip(rows, cols) if matrix[r, c] >= self.threshold]


def build_matchers(seed: int = 0) -> dict[str, object]:
    learned = LearnedMatcher()
    forest = LearnedMatcher("m3_forest", RandomForestClassifier(n_estimators=200, random_state=seed,
                                                               n_jobs=-1, class_weight="balanced"))
    matchers = (ExactMatcher(), FuzzyMatcher(), learned, forest, AssignmentMatcher(learned))
    return {m.name: m for m in matchers}
