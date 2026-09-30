"""Scoring anomaly detection under heavy class imbalance.

Accuracy is meaningless here: flagging nothing scores about 96%. Precision-recall summaries are
reported instead, with the false-positive rate, because every false positive costs review time.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_fscore_support, roc_auc_score


def summarise(labels: np.ndarray, scores: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    predicted = (scores >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predicted, average="binary", zero_division=0)
    negatives = int((labels == 0).sum())
    false_positives = int(((predicted == 1) & (labels == 0)).sum())
    single_class = len(set(labels.tolist())) < 2
    return {
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1": round(float(f1), 4),
        "false_positive_rate": round(false_positives / negatives, 4) if negatives else 0.0,
        "flagged_rate": round(float(predicted.mean()), 4),
        "roc_auc": round(float(roc_auc_score(labels, scores)), 4) if not single_class else float("nan"),
        "average_precision": round(float(average_precision_score(labels, scores)), 4)
        if not single_class else float("nan"),
    }
