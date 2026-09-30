"""Four visually distinct invoice layouts.

Field positions, fonts, column orders and tax presentation all differ, so an extractor tuned to one
template does not transfer to the others.
"""

from __future__ import annotations

from collections.abc import Callable
from decimal import Decimal

from gstpipe.gst import STATES
from gstpipe.render.layout import Pen, inr
from gstpipe.render.tables import Col, draw_table, qty
from gstpipe.render.words import rupees_in_words
from gstpipe.schema import Invoice

L, R = 36.0, 559.0  # page margins in points


def _intra(inv: Invoice) -> bool:
    return inv.supplier.state_code == inv.place_of_supply


def _pos(inv: Invoice) -> str:
    return f"{inv.place_of_supply} - {STATES[inv.place_of_supply][0]}"


def _meta(inv: Invoice) -> list[tuple[str, str]]:
    return [("Invoice No", inv.invoice_number),
            ("Invoice Date", inv.invoice_date.strftime("%d-%m-%Y")),
            ("Due Date", inv.due_date.strftime("%d-%m-%Y")),
            ("Place of Supply", _pos(inv))]


def _totals(inv: Invoice) -> list[tuple[str, Decimal]]:
    rows = [("Taxable Value", inv.taxable_total)]
    if _intra(inv):
        rows += [("CGST", inv.cgst_total), ("SGST", inv.sgst_total)]
    else:
        rows += [("IGST", inv.igst_total)]
    if inv.round_off:
        rows.append(("Round Off", inv.round_off))
    return rows


def _rows(inv: Invoice, *, split_tax: bool) -> list[dict[str, str]]:
    out = []
    for li in inv.line_items:
        row = {"sr": str(li.line_no), "desc": li.description, "hsn": li.hsn_sac, "qty": qty(li.quantity),
               "rate": inr(li.unit_price), "taxable": inr(li.taxable_value), "gst": f"{li.gst_rate:g}%"}
        if split_tax:
            row |= ({"cgst": inr(li.cgst), "sgst": inr(li.sgst)} if _intra(inv) else {"igst": inr(li.igst)})
        else:
            row["tax"] = inr(li.cgst + li.sgst + li.igst)
        out.append(row)
    return out


# --------------------------------------------------------------------------- classic

def classic(pen: Pen, inv: Invoice) -> None:
    pen.font, pen.regular = "Helvetica", ""
    pen.rect(L, 30, R - L, 760)
    pen.centre((L + R) / 2, 52, "TAX INVOICE", "-Bold", 13)
    pen.hline(L, 62, R)

    y = pen.block(L + 8, 80, [inv.supplier.name], "-Bold", 11)
    y = pen.block(L + 8, y + 2, pen.wrap(inv.supplier.address, 250, size=8.5), size=8.5, leading=11)
    pen.text(L + 8, y + 2, f"GSTIN: {inv.supplier.gstin}", "-Bold", 9)
    pen.text(L + 8, y + 14, f"State: {STATES[inv.supplier.state_code][0]} ({inv.supplier.state_code})", size=8.5)

    pen.vline(330, 66, 150)
    for i, (label, value) in enumerate(_meta(inv)):
        pen.text(340, 82 + i * 15, label, size=8.5)
        pen.text(440, 82 + i * 15, value, "-Bold", 8.5)
    pen.hline(L, 150, R)

    pen.text(L + 8, 166, "BILL TO", "-Bold", 8.5)
    y = pen.block(L + 8, 180, [inv.recipient.name], "-Bold", 10)
    y = pen.block(L + 8, y, pen.wrap(inv.recipient.address, 300, size=8.5), size=8.5, leading=11)
    pen.text(L + 8, y, f"GSTIN: {inv.recipient.gstin}", size=8.5)

    cols = [Col("sr", "Sr", L, 26, "c"), Col("desc", "Description", L + 26, 190),
            Col("hsn", "HSN/SAC", L + 216, 58, "c"), Col("qty", "Qty", L + 274, 38, "r"),
            Col("rate", "Rate", L + 312, 62, "r"), Col("taxable", "Taxable", L + 374, 70, "r")]
    cols += ([Col("cgst", "CGST", L + 444, 40, "r"), Col("sgst", "SGST", L + 484, 39, "r")]
             if _intra(inv) else [Col("igst", "IGST", L + 444, 79, "r")])
    y = draw_table(pen, cols, _rows(inv, split_tax=True), 250, grid=True)

    for i, (label, value) in enumerate(_totals(inv)):
        pen.right(470, y + 16 + i * 14, label, size=9)
        pen.right(R - 6, y + 16 + i * 14, inr(value), size=9)
    bar = y + 16 + len(_totals(inv)) * 14
    pen.hline(370, bar - 9, R, 0.8)
    pen.right(470, bar + 4, "TOTAL", "-Bold", 10.5)
    pen.right(R - 6, bar + 4, inr(inv.invoice_total), "-Bold", 10.5)

    pen.text(L + 8, bar + 30, "Amount in words:", "-Bold", 8.5)
    pen.block(L + 8, bar + 42, pen.wrap(rupees_in_words(inv.invoice_total), 480, size=8.5), size=8.5, leading=11)
    pen.text(L + 8, 740, "Terms: Goods once sold will not be taken back. Interest @18% p.a. on overdue bills.", size=7.5)
    pen.right(R - 8, 730, f"For {inv.supplier.name}", "-Bold", 9)
    pen.right(R - 8, 775, "Authorised Signatory", size=8.5)


# --------------------------------------------------------------------------- modern

def modern(pen: Pen, inv: Invoice) -> None:
    pen.font, pen.regular = "Helvetica", ""
    pen.band(0, 0, 595, 6, (0.16, 0.36, 0.66))
    pen.text(L, 56, inv.supplier.name, "-Bold", 17)
    y = pen.block(L, 72, pen.wrap(inv.supplier.address, 260, size=8.5), size=8.5, leading=11)
    pen.text(L, y + 2, f"GSTIN {inv.supplier.gstin}", size=8.5)

    pen.right(R, 58, "INVOICE", "-Bold", 22)
    for i, (label, value) in enumerate(_meta(inv)):
        pen.right(468, 84 + i * 14, label.upper(), size=7.5)
        pen.right(R, 84 + i * 14, value, "-Bold", 9)

    pen.hline(L, 168, R, 0.8, 0.75)
    pen.text(L, 186, "BILLED TO", "-Bold", 7.5)
    y = pen.block(L, 200, [inv.recipient.name], "-Bold", 10.5)
    y = pen.block(L, y, pen.wrap(inv.recipient.address, 300, size=8.5), size=8.5, leading=11)
    pen.text(L, y, f"GSTIN {inv.recipient.gstin}   |   Place of Supply: {_pos(inv)}", size=8.5)

    cols = [Col("desc", "DESCRIPTION", L, 214), Col("hsn", "HSN/SAC", L + 214, 56, "c"),
            Col("qty", "QTY", L + 270, 40, "r"), Col("rate", "RATE", L + 310, 66, "r"),
            Col("gst", "GST", L + 376, 40, "c"), Col("tax", "TAX", L + 416, 50, "r"),
            Col("taxable", "TAXABLE", L + 466, 57, "r")]
    y = draw_table(pen, cols, _rows(inv, split_tax=False), 262, zebra=True, rules=False,
                   header_fill=(0.90, 0.93, 0.98), row_h=16)

    for i, (label, value) in enumerate(_totals(inv)):
        pen.right(480, y + 20 + i * 14, label, size=9)
        pen.right(R, y + 20 + i * 14, inr(value), size=9)
    bar = y + 20 + len(_totals(inv)) * 14
    pen.band(360, bar - 9, R - 360, 22, (0.90, 0.93, 0.98))
    pen.right(480, bar + 6, "AMOUNT DUE", "-Bold", 10)
    pen.right(R, bar + 6, inr(inv.invoice_total), "-Bold", 11)
    pen.block(L, bar + 40, pen.wrap(rupees_in_words(inv.invoice_total), 300, size=8), size=8, leading=10.5)
    pen.hline(L, 752, R, 0.6, 0.8)
    pen.text(L, 766, "Bank: HDFC Bank  |  A/c 50200034567891  |  IFSC HDFC0000123", size=7.5)


# --------------------------------------------------------------------------- compact

def compact(pen: Pen, inv: Invoice) -> None:
    pen.font, pen.regular = "Courier", ""
    pen.text(L, 48, inv.supplier.name.upper(), "-Bold", 11)
    pen.text(L, 62, inv.supplier.address, size=7.5)
    pen.text(L, 74, f"GSTIN {inv.supplier.gstin}  STATE {inv.supplier.state_code}", size=7.5)
    pen.hline(L, 82, R, 0.8)
    pen.centre((L + R) / 2, 96, "TAX INVOICE", "-Bold", 10)

    for i, (label, value) in enumerate(_meta(inv)):
        pen.text(L, 114 + i * 12, f"{label.upper():<16}: {value}", size=8)
    pen.text(320, 114, "BILLED TO:", "-Bold", 8)
    pen.block(320, 126, [*pen.wrap(inv.recipient.name, 225, size=7.5),
                         *pen.wrap(inv.recipient.address, 225, size=7.5),
                         f"GSTIN {inv.recipient.gstin}"], size=7.5, leading=10)
    pen.hline(L, 176, R, 0.8)

    cols = [Col("sr", "SR", L, 22), Col("desc", "ITEM", L + 22, 196), Col("hsn", "HSN", L + 218, 50),
            Col("qty", "QTY", L + 268, 34, "r"), Col("rate", "RATE", L + 302, 58, "r"),
            Col("gst", "GST", L + 360, 34, "r"), Col("taxable", "VALUE", L + 394, 64, "r"),
            Col("tax", "TAX", L + 458, 65, "r")]
    y = draw_table(pen, cols, _rows(inv, split_tax=False), 186, row_h=13, header_h=14, size=7.5)

    for i, (label, value) in enumerate(_totals(inv)):
        pen.text(370, y + 14 + i * 12, f"{label.upper():<14}", size=8)
        pen.right(R, y + 14 + i * 12, inr(value), size=8)
    bar = y + 14 + len(_totals(inv)) * 12
    pen.hline(360, bar - 8, R, 0.6)
    pen.text(370, bar + 4, "GRAND TOTAL", "-Bold", 9)
    pen.right(R, bar + 4, inr(inv.invoice_total), "-Bold", 9)
    pen.block(L, bar + 26, pen.wrap("RS. " + rupees_in_words(inv.invoice_total).upper(), 500, size=7.5),
              size=7.5, leading=10)
    pen.hline(L, 770, R, 0.6)
    pen.text(L, 782, "E.& O.E.   This is a computer generated invoice.", size=7)


# --------------------------------------------------------------------------- letterhead

def letterhead(pen: Pen, inv: Invoice) -> None:
    pen.font, pen.regular = "Times", "-Roman"
    pen.band(0, 0, 595, 62, (0.10, 0.22, 0.38))
    pen.c.setFillColorRGB(1, 1, 1)
    pen.set("-Bold", 18)
    pen.c.drawString(L, 842 - 38, inv.supplier.name)
    pen.set("-Roman", 8.5)
    pen.c.drawString(L, 842 - 52, f"{inv.supplier.address}   |   GSTIN {inv.supplier.gstin}")
    pen.c.setFillGray(0)

    pen.centre((L + R) / 2, 92, "TAX INVOICE", "-Bold", 12)
    pen.hline(240, 98, 355, 0.8)

    pen.text(L, 126, "Invoice to", "-Bold", 9)
    y = pen.block(L, 142, [inv.recipient.name], "-Bold", 11)
    y = pen.block(L, y, pen.wrap(inv.recipient.address, 270, size=9), size=9, leading=12)
    pen.text(L, y, f"GSTIN {inv.recipient.gstin}", size=9)
    for i, (label, value) in enumerate(_meta(inv)):
        pen.text(360, 142 + i * 15, f"{label}:", size=9)
        pen.right(R, 142 + i * 15, value, "-Bold", 9)

    cols = [Col("sr", "#", L, 20), Col("desc", "Particulars", L + 20, 206),
            Col("hsn", "HSN/SAC", L + 226, 56, "c"), Col("qty", "Qty", L + 282, 36, "r"),
            Col("rate", "Rate", L + 318, 64, "r"), Col("gst", "GST", L + 382, 38, "c"),
            Col("taxable", "Amount", L + 420, 103, "r")]
    y = draw_table(pen, cols, _rows(inv, split_tax=False), 232, row_h=16, size=9)

    pen.rect(350, y + 10, R - 350, 18 + len(_totals(inv)) * 14, 0.5, 0.55)
    for i, (label, value) in enumerate(_totals(inv)):
        pen.text(358, y + 26 + i * 14, label, size=9)
        pen.right(R - 6, y + 26 + i * 14, inr(value), size=9)
    bar = y + 28 + len(_totals(inv)) * 14
    pen.text(358, bar + 14, "Total Payable", "-Bold", 10.5)
    pen.right(R - 6, bar + 14, inr(inv.invoice_total), "-Bold", 10.5)
    pen.text(L, y + 26, "Amount in words", "-Bold", 8.5)
    pen.block(L, y + 40, pen.wrap(rupees_in_words(inv.invoice_total), 300, size=8.5), size=8.5, leading=11)
    pen.band(0, 800, 595, 42, (0.93, 0.94, 0.96))
    pen.text(L, 818, "Declaration: We declare that this invoice shows the actual price of the goods described.", size=7.5)
    pen.right(R, 828, f"For {inv.supplier.name}", "-Bold", 9)


TEMPLATES: dict[str, Callable[[Pen, Invoice], None]] = {
    "classic": classic, "modern": modern, "compact": compact, "letterhead": letterhead,
}
