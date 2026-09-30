"""Drawing helpers shared by the invoice templates."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen.canvas import Canvas

PAGE_W, PAGE_H = A4


def inr(amount: Decimal | float, decimals: bool = True) -> str:
    """Indian digit grouping: 1234567.5 -> '12,34,567.50'."""
    sign = "-" if amount < 0 else ""
    whole, _, frac = f"{abs(Decimal(str(amount))):.2f}".partition(".")
    if len(whole) > 3:
        head, tail = whole[:-3], whole[-3:]
        groups = []
        while len(head) > 2:
            head, group = head[:-2], head[-2:]
            groups.insert(0, group)
        whole = ",".join(filter(None, [head, *groups, tail]))
    return f"{sign}{whole}.{frac}" if decimals else f"{sign}{whole}"


@dataclass
class Pen:
    """Thin wrapper over the PDF canvas: y grows downward from the top margin."""

    c: Canvas
    font: str = "Helvetica"
    regular: str = ""
    size: float = 9.0
    leading: float = 12.0

    def set(self, style: str = "", size: float | None = None) -> None:
        self.c.setFont(f"{self.font}{style or self.regular}", size or self.size)

    def text(self, x: float, y: float, s: str, style: str = "", size: float | None = None) -> None:
        self.set(style, size)
        self.c.drawString(x, PAGE_H - y, s)

    def right(self, x: float, y: float, s: str, style: str = "", size: float | None = None) -> None:
        self.set(style, size)
        self.c.drawRightString(x, PAGE_H - y, s)

    def centre(self, x: float, y: float, s: str, style: str = "", size: float | None = None) -> None:
        self.set(style, size)
        self.c.drawCentredString(x, PAGE_H - y, s)

    def block(self, x: float, y: float, lines, style: str = "", size: float | None = None,
              leading: float | None = None) -> float:
        step = leading or self.leading
        for i, line in enumerate(lines):
            self.text(x, y + i * step, line, style, size)
        return y + len(lines) * step

    def hline(self, x1: float, y: float, x2: float, width: float = 0.6, grey: float = 0.0) -> None:
        self.c.setLineWidth(width)
        self.c.setStrokeGray(grey)
        self.c.line(x1, PAGE_H - y, x2, PAGE_H - y)

    def vline(self, x: float, y1: float, y2: float, width: float = 0.6, grey: float = 0.0) -> None:
        self.c.setLineWidth(width)
        self.c.setStrokeGray(grey)
        self.c.line(x, PAGE_H - y1, x, PAGE_H - y2)

    def rect(self, x: float, y: float, w: float, h: float, width: float = 0.6, grey: float = 0.0,
             fill: tuple[float, float, float] | None = None) -> None:
        self.c.setLineWidth(width)
        self.c.setStrokeGray(grey)
        if fill:
            self.c.setFillColorRGB(*fill)
        self.c.rect(x, PAGE_H - y - h, w, h, stroke=1, fill=1 if fill else 0)
        self.c.setFillGray(0)

    def band(self, x: float, y: float, w: float, h: float, fill: tuple[float, float, float]) -> None:
        self.c.setFillColorRGB(*fill)
        self.c.rect(x, PAGE_H - y - h, w, h, stroke=0, fill=1)
        self.c.setFillGray(0)

    def wrap(self, s: str, width: float, style: str = "", size: float | None = None) -> list[str]:
        """Greedy wrap of a single string to the given width in points."""
        font = f"{self.font}{style or self.regular}"
        pt = size or self.size
        words, lines, current = s.split(), [], ""
        for word in words:
            trial = f"{current} {word}".strip()
            if self.c.stringWidth(trial, font, pt) <= width or not current:
                current = trial
            else:
                lines.append(current)
                current = word
        if current:
            lines.append(current)
        return lines
