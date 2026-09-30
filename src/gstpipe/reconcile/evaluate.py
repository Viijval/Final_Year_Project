"""Scoring reconciliation.

A false match is worse than a missed one in accounting: it silently closes the wrong invoice. The
two are therefore reported separately, along with how much work is left for a human.
"""

from __future__ import annotations

from dataclasses import dataclass

from gstpipe.reconcile.matchers import Prediction


@dataclass
class ReconResult:
    true_pairs: int
    predicted: int
    correct: int
    false_matches: int
    txns_to_match: int
    txns_matched: int
    review: int

    @property
    def precision(self) -> float:
        return self.correct / self.predicted if self.predicted else 0.0

    @property
    def recall(self) -> float:
        return self.correct / self.true_pairs if self.true_pairs else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if p + r else 0.0

    @property
    def false_match_rate(self) -> float:
        return self.false_matches / self.predicted if self.predicted else 0.0

    @property
    def unmatched_rate(self) -> float:
        return 1 - (self.txns_matched / self.txns_to_match) if self.txns_to_match else 0.0

    @property
    def review_rate(self) -> float:
        return self.review / self.predicted if self.predicted else 0.0

    def as_row(self) -> dict[str, float]:
        return {"precision": round(self.precision, 4), "recall": round(self.recall, 4),
                "f1": round(self.f1, 4), "false_match_rate": round(self.false_match_rate, 4),
                "unmatched_rate": round(self.unmatched_rate, 4),
                "review_rate": round(self.review_rate, 4), "predicted": self.predicted,
                "correct": self.correct}


def evaluate(predictions: list[Prediction], truth: set[tuple[str, str]],
             payable_txns: set[str]) -> ReconResult:
    predicted_pairs = {(p.txn_id, p.invoice_id) for p in predictions}
    correct = len(predicted_pairs & truth)
    return ReconResult(
        true_pairs=len(truth),
        predicted=len(predicted_pairs),
        correct=correct,
        false_matches=len(predicted_pairs - truth),
        txns_to_match=len(payable_txns),
        txns_matched=len({t for t, _ in predicted_pairs & truth}),
        review=sum(p.needs_review for p in predictions),
    )
