"""Three detection tiers: deterministic checks, per-supplier statistics, and unsupervised models.

Every detector returns a continuous score so the precision-recall trade-off is visible, rather than
a single operating point chosen in advance. In this setting a false positive costs professional
review time and a false negative lets an error through, and those costs are not symmetric.
"""

from __future__ import annotations

from collections import defaultdict
from statistics import median

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

from gstpipe.anomaly.features import InvoiceSignals

MAD_TO_SIGMA = 1.4826  # scales median absolute deviation to a standard-deviation equivalent


class RuleDetector:
    """A1: the document contradicts itself, or repeats a number the supplier already used."""

    name = "a1_rules"

    def fit(self, signals: list[InvoiceSignals]) -> RuleDetector:
        return self

    def score(self, signals: list[InvoiceSignals]) -> np.ndarray:
        return np.array([min(1.0, len(s.structural_failures) / 2 + 0.5 * bool(s.structural_failures))
                         for s in signals])


class StatisticalDetector:
    """A2: a robust z-score computed within each supplier, not across the whole ledger."""

    name = "a2_statistical"

    def __init__(self, floor: float = 3.0):
        self.floor = floor
        self.centre: dict[str, float] = {}
        self.spread: dict[str, float] = {}

    def fit(self, signals: list[InvoiceSignals]) -> StatisticalDetector:
        grouped: dict[str, list[float]] = defaultdict(list)
        for signal in signals:
            if signal.amount is not None:
                grouped[signal.supplier_key].append(float(signal.amount))
        for key, values in grouped.items():
            mid = median(values)
            deviation = median([abs(v - mid) for v in values]) * MAD_TO_SIGMA
            self.centre[key] = mid
            self.spread[key] = deviation if deviation > 1e-6 else max(1.0, mid * 0.1)
        return self

    def score(self, signals: list[InvoiceSignals]) -> np.ndarray:
        out = []
        for signal in signals:
            amount = float(signal.amount) if signal.amount is not None else None
            centre = self.centre.get(signal.supplier_key)
            if amount is None or centre is None:
                # Unseen supplier: fall back to how far above its own recorded median this sits.
                out.append(min(1.0, max(0.0, (signal.features["amount_over_vendor_median"] - 1) / 8)))
                continue
            z = abs(amount - centre) / self.spread[signal.supplier_key]
            out.append(min(1.0, z / (self.floor * 3)))
        return np.array(out)


class ModelDetector:
    """A3: unsupervised multivariate detection over the whole signal vector."""

    def __init__(self, name: str, estimator):
        self.name = name
        self.estimator = estimator
        self.scaler = StandardScaler()

    def fit(self, signals: list[InvoiceSignals]) -> ModelDetector:
        matrix = self.scaler.fit_transform(np.array([s.vector for s in signals]))
        self.estimator.fit(matrix)
        return self

    def score(self, signals: list[InvoiceSignals]) -> np.ndarray:
        matrix = self.scaler.transform(np.array([s.vector for s in signals]))
        raw = -self.estimator.decision_function(matrix)  # higher means more anomalous
        lo, hi = raw.min(), raw.max()
        return (raw - lo) / (hi - lo) if hi > lo else np.zeros_like(raw)


def build_detectors(seed: int = 0, contamination: float = 0.05) -> dict[str, object]:
    detectors = (
        RuleDetector(),
        StatisticalDetector(),
        ModelDetector("a3_isolation_forest",
                      IsolationForest(n_estimators=300, contamination=contamination,
                                      random_state=seed, n_jobs=-1)),
        ModelDetector("a3_lof", LocalOutlierFactor(n_neighbors=25, contamination=contamination,
                                                   novelty=True)),
    )
    return {d.name: d for d in detectors}
