"""A page of positioned words, so extractors can use layout and not just raw text.

Both OCR and the PDF text layer produce this, which lets the same extractor run on a perfect
text source (an upper bound) and on a degraded scan (the realistic case).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Word:
    text: str
    x0: float
    y0: float
    x1: float
    y1: float
    conf: float = 1.0

    @property
    def cx(self) -> float:
        return (self.x0 + self.x1) / 2

    @property
    def cy(self) -> float:
        return (self.y0 + self.y1) / 2


@dataclass
class Line:
    words: list[Word]

    @property
    def text(self) -> str:
        return " ".join(w.text for w in self.words)

    @property
    def y(self) -> float:
        return sum(w.cy for w in self.words) / len(self.words)

    @property
    def x0(self) -> float:
        return min(w.x0 for w in self.words)

    def after(self, x: float) -> list[Word]:
        return [w for w in self.words if w.x0 >= x]


@dataclass
class Page:
    words: list[Word]
    width: float
    height: float
    _lines: list[Line] | None = field(default=None, repr=False)

    @property
    def lines(self) -> list[Line]:
        """Group words into visual lines by vertical proximity."""
        if self._lines is None:
            tolerance = max(4.0, self.height * 0.006)
            grouped: list[Line] = []
            for word in sorted(self.words, key=lambda w: (w.cy, w.x0)):
                if grouped and abs(grouped[-1].y - word.cy) <= tolerance:
                    grouped[-1].words.append(word)
                else:
                    grouped.append(Line([word]))
            for line in grouped:
                line.words.sort(key=lambda w: w.x0)
            self._lines = grouped
        return self._lines

    @property
    def text(self) -> str:
        return "\n".join(line.text for line in self.lines)

    def find(self, pattern: str) -> list[tuple[Line, re.Match[str]]]:
        """Lines whose text matches a case-insensitive pattern, in reading order."""
        rx = re.compile(pattern, re.IGNORECASE)
        return [(line, m) for line in self.lines if (m := rx.search(line.text))]

    def scale_to(self, width: float, height: float) -> Page:
        """Normalise coordinates, so rules written against one resolution work at any DPI."""
        sx, sy = width / self.width, height / self.height
        return Page([Word(w.text, w.x0 * sx, w.y0 * sy, w.x1 * sx, w.y1 * sy, w.conf) for w in self.words],
                    width, height)
