"""Flat and hierarchy-aware scoring for categorisation.

Predicting "Technology > Software Licences" when the truth is "Technology > Cloud Services" is a
better outcome than predicting "Payroll", and level accuracy shows that; flat accuracy cannot.
"""

from __future__ import annotations

from sklearn.metrics import f1_score

SEPARATOR = " > "
LEVELS = ("l1", "l2", "l3", "leaf")


def levels(path: str) -> list[str]:
    return path.split(SEPARATOR)


def level_accuracy(truth: list[str], predicted: list[str]) -> dict[str, float]:
    """Accuracy at each depth: agreement on the prefix down to that level."""
    scores = {}
    for depth, name in enumerate(LEVELS, start=1):
        hits = sum(levels(t)[:depth] == levels(p)[:depth] for t, p in zip(truth, predicted))
        scores[name] = round(hits / len(truth), 4) if truth else 0.0
    return scores


def summarise(truth: list[str], predicted: list[str]) -> dict[str, float]:
    accuracy = sum(t == p for t, p in zip(truth, predicted)) / len(truth) if truth else 0.0
    return {
        "accuracy": round(accuracy, 4),
        "macro_f1": round(f1_score(truth, predicted, average="macro", zero_division=0), 4),
        "weighted_f1": round(f1_score(truth, predicted, average="weighted", zero_division=0), 4),
        **level_accuracy(truth, predicted),
    }
