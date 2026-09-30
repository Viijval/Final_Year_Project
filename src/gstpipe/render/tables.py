"""A configurable items table so templates can share code while still looking different."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from gstpipe.render.layout import Pen


@dataclass
class Col:
    key: str
    header: str
    x: float
    w: float
    align: str = "l"  # l, r or c


def qty(value: Decimal) -> str:
    return str(int(value)) if value == value.to_integral_value() else f"{value:f}"


def draw_table(pen: Pen, cols: list[Col], rows: list[dict[str, str]], y: float, *, row_h: float = 15,
               header_h: float = 16, grid: bool = False, zebra: bool = False,
               header_fill: tuple[float, float, float] | None = None, rules: bool = True,
               size: float = 8.5) -> float:
    x0, x1 = cols[0].x, cols[-1].x + cols[-1].w
    if header_fill:
        pen.band(x0, y, x1 - x0, header_h, header_fill)
    for col in cols:
        _cell(pen, col, col.header, y + header_h - 5, "-Bold", size)
    y += header_h
    if rules:
        pen.hline(x0, y, x1, 0.8)

    for i, row in enumerate(rows):
        if zebra and i % 2:
            pen.band(x0, y, x1 - x0, row_h, (0.955, 0.965, 0.98))
        for col in cols:
            _cell(pen, col, row.get(col.key, ""), y + row_h - 5, "", size)
        y += row_h
        if grid:
            pen.hline(x0, y, x1, 0.4, 0.6)
    if rules:
        pen.hline(x0, y, x1, 0.8)
    if grid:
        for col in cols:
            pen.vline(col.x, y - row_h * len(rows) - header_h, y, 0.4, 0.6)
        pen.vline(x1, y - row_h * len(rows) - header_h, y, 0.4, 0.6)
    return y


def _cell(pen: Pen, col: Col, text: str, y: float, style: str, size: float) -> None:
    if col.align == "r":
        pen.right(col.x + col.w - 4, y, text, style, size)
    elif col.align == "c":
        pen.centre(col.x + col.w / 2, y, text, style, size)
    else:
        pen.text(col.x + 4, y, text, style, size)
