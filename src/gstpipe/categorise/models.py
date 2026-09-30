"""Tier C2: supervised text models over different representations.

The tiers are chosen to answer one question: does a representation that degrades gracefully under
character corruption (char n-grams) survive OCR noise better than a word-level one, and how much do
numeric features -- which OCR cannot touch -- make up the difference?
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MaxAbsScaler

from gstpipe.categorise.features import CategoryInput


@dataclass
class Representation:
    words: bool = True
    chars: bool = False
    numeric: bool = False

    def build(self):
        parts = {}
        if self.words:
            parts["word"] = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
        if self.chars:
            parts["char"] = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2,
                                            sublinear_tf=True)
        return parts


class Vectoriser:
    def __init__(self, representation: Representation):
        self.representation = representation
        self.text_parts = representation.build()
        self.numeric = DictVectorizer(sparse=True) if representation.numeric else None
        self.scaler = MaxAbsScaler() if representation.numeric else None

    def _matrix(self, items: list[CategoryInput], fit: bool):
        texts = [item.text for item in items]
        blocks = []
        for vec in self.text_parts.values():
            blocks.append(vec.fit_transform(texts) if fit else vec.transform(texts))
        if self.numeric is not None:
            rows = [item.numeric for item in items]
            numeric = self.numeric.fit_transform(rows) if fit else self.numeric.transform(rows)
            numeric = self.scaler.fit_transform(numeric) if fit else self.scaler.transform(numeric)
            blocks.append(csr_matrix(numeric))
        return hstack(blocks).tocsr() if len(blocks) > 1 else blocks[0]

    def fit_transform(self, items):
        return self._matrix(items, fit=True)

    def transform(self, items):
        return self._matrix(items, fit=False)


class Categoriser:
    def __init__(self, name: str, representation: Representation, estimator):
        self.name = name
        self.vectoriser = Vectoriser(representation)
        self.estimator = estimator

    def fit(self, items: list[CategoryInput], labels: list[str]) -> Categoriser:
        self.estimator.fit(self.vectoriser.fit_transform(items), np.asarray(labels))
        return self

    def predict(self, items: list[CategoryInput]) -> list[str]:
        return list(self.estimator.predict(self.vectoriser.transform(items)))


def build_models(seed: int = 0) -> dict[str, Categoriser]:
    return {
        "word_lr": Categoriser("word_lr", Representation(words=True),
                               LogisticRegression(max_iter=2000, C=4.0)),
        "char_lr": Categoriser("char_lr", Representation(words=False, chars=True),
                               LogisticRegression(max_iter=2000, C=4.0)),
        "hybrid_lr": Categoriser("hybrid_lr", Representation(words=True, chars=True, numeric=True),
                                 LogisticRegression(max_iter=2000, C=4.0)),
        "hybrid_rf": Categoriser("hybrid_rf", Representation(words=True, chars=True, numeric=True),
                                 RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=-1)),
    }
