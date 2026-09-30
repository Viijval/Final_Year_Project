"""Tier 1 extractor: label-anchored regular expressions over positioned words.

Deliberately not robust. It leans on labels, reading order and column gaps, so it does well on
layouts it was written against and degrades on unfamiliar ones and on noisy OCR. That contrast is
the baseline the later tiers are measured against.
"""

from __future__ import annotations

import re
from datetime import date, datetime
from decimal import Decimal, InvalidOperation

from gstpipe.extract.page import Line, Page, Word
from gstpipe.extract.records import ExtractedInvoice, ExtractedLine, Region

MONEY = re.compile(r"-?\d{1,3}(?:,\d{2,3})*\.\d{2}|-?\d+\.\d{2}|-?\d{1,3}(?:,\d{2,3})+")
DATE = re.compile(r"(\d{1,2})[-/](\d{1,2})[-/](\d{2,4})")
GSTIN = re.compile(r"\b[0-9A-Z]{15}\b")
HSN = re.compile(r"\b\d{4,8}\b")
PERCENT = re.compile(r"(\d{1,2}(?:\.\d+)?)\s*%")
HEADING = re.compile(r"^(tax\s+invoice|invoice|bill\s+of\s+supply)$", re.IGNORECASE)

COLUMN_GAP = 0.055  # a horizontal gap wider than this share of the page starts a new column


def _money(text: str) -> Decimal | None:
    try:
        return Decimal(text.replace(",", ""))
    except (InvalidOperation, AttributeError):
        return None


def _last_money(text: str) -> Decimal | None:
    found = MONEY.findall(text)
    return _money(found[-1]) if found else None


def _date(text: str) -> date | None:
    if not (m := DATE.search(text)):
        return None
    day, month, year = (int(g) for g in m.groups())
    year += 2000 if year < 100 else 0
    for d, mo in ((day, month), (month, day)):  # tolerate a swapped pair only if the first is impossible
        try:
            return datetime(year, mo, d).date()
        except ValueError:
            continue
    return None


def _words_from(words: list[Word], page: Page) -> list[Word]:
    """Take words until a column-sized horizontal gap appears."""
    if not words:
        return []
    gap = page.width * COLUMN_GAP
    kept = [words[0]]
    for previous, word in zip(words, words[1:]):
        if word.x0 - previous.x1 > gap:
            break
        kept.append(word)
    return kept


def _split_at(line: Line, index: int) -> tuple[list[Word], list[Word]]:
    return line.words[:index], line.words[index:]


def _region(words: list[Word], page: Page) -> Region | None:
    """The box enclosing the words a value was read from."""
    if not words:
        return None
    return Region(x0=min(w.x0 for w in words), y0=min(w.y0 for w in words),
                  x1=max(w.x1 for w in words), y1=max(w.y1 for w in words),
                  text=" ".join(w.text for w in words),
                  page_width=page.width, page_height=page.height)


def _labelled(page: Page, *labels: str) -> list[tuple[Line, int, str, list[Word]]]:
    """Lines carrying a label, with the word index where the label starts and the value beside it.

    The value stops at the first column gap, so a neighbouring column cannot bleed into it.
    """
    out = []
    for label in labels:
        for line, match in page.find(label):
            head = len(line.text[:match.start()].split())
            consumed = len(line.text[:match.end()].split())
            _, rest = _split_at(line, consumed)
            words = _words_from(rest, page)
            value = " ".join(w.text for w in words).strip(" :.-|")
            out.append((line, head, value, words))
    return out


def _value(page: Page, *labels: str) -> tuple[str | None, list[Word]]:
    for _, _, value, words in _labelled(page, *labels):
        if value:
            return value, words
    return None, []


def _amount(page: Page, *labels: str) -> tuple[Decimal | None, list[Word]]:
    for line, _, value, words in _labelled(page, *labels):
        if (amount := _last_money(value)) is not None:
            return amount, words
        if (amount := _last_money(line.text)) is not None:
            return amount, line.words
    return None, []


BILL_TO = re.compile(r"bill\s*to|billed\s*to|invoice\s*to|buyer|customer", re.IGNORECASE)


def _gstin_positions(page: Page) -> list[tuple[str, Word]]:
    found: list[tuple[float, str, Word]] = []
    seen: set[str] = set()
    for line in page.lines:
        for word in line.words:
            token = word.text.strip(" :,|")
            if (GSTIN.fullmatch(token) and any(c.isdigit() for c in token)
                    and any(c.isalpha() for c in token) and token not in seen):
                seen.add(token)
                found.append((word.cy, token, word))
    return [(token, word) for _, token, word in sorted(found, key=lambda f: f[0])]


def _gstins(page: Page) -> list[str]:
    """GSTIN-shaped tokens in reading order."""
    return [token for token, _ in _gstin_positions(page)]


def _gstins_by_role(page: Page) -> tuple[tuple[str | None, list[Word]], tuple[str | None, list[Word]]]:
    """Assign GSTINs to supplier and customer by which side of the "bill to" heading they sit on.

    Reading order alone is not enough: OCR frequently loses one of the two, and assuming the
    survivor is the supplier silently swaps the parties. Every layout prints the customer block
    under a heading, so position relative to that heading decides the role.
    """
    found = _gstin_positions(page)
    if not found:
        return (None, []), (None, [])
    anchors = [line.y for line in page.lines if BILL_TO.search(line.text)]
    if not anchors:
        second = (found[1][0], [found[1][1]]) if len(found) > 1 else (None, [])
        return (found[0][0], [found[0][1]]), second
    boundary = min(anchors)
    above = [(token, word) for token, word in found if word.cy < boundary]
    below = [(token, word) for token, word in found if word.cy >= boundary]
    return ((above[0][0], [above[0][1]]) if above else (None, []),
            (below[0][0], [below[0][1]]) if below else (None, []))


def _supplier_name(page: Page) -> tuple[str | None, list[Word]]:
    """Top-left block. Metadata often shares the same visual line, so stop at the column gap."""
    for line in page.lines[:6]:
        words = _words_from(line.words, page)
        text = " ".join(w.text for w in words).strip(" :|")
        if len(text) > 3 and not HEADING.match(text) and not GSTIN.fullmatch(text.replace(" ", "")):
            if sum(c.isalpha() for c in text) >= len(text) * 0.5:
                return text, words
    return None, []


def _recipient_name(page: Page) -> tuple[str | None, list[Word]]:
    anchors = (r"bill\s*to", r"billed\s*to", r"invoice\s*to", r"buyer", r"customer")
    for line, head, value, words in _labelled(page, *anchors):
        if value:
            return value, words
        anchor_x = line.words[head].x0 if head < len(line.words) else line.x0
        index = page.lines.index(line)
        for following in page.lines[index + 1:index + 3]:
            column = [w for w in following.words if w.x0 >= anchor_x - page.width * 0.02]
            kept = _words_from(column, page)
            text = " ".join(w.text for w in kept).strip(" :|")
            if text and not re.match(r"^(gstin|state|place)", text, re.IGNORECASE):
                return text, kept
    return None, []


def _hsn_column(page: Page) -> float | None:
    """Centre of the HSN/SAC column, so digits inside a description are not mistaken for a code."""
    for line in page.lines:
        for word in line.words:
            if re.fullmatch(r"hsn(/sac)?|sac|hsn/?", word.text.strip(), re.IGNORECASE):
                return word.cx
    return None


def _line_items(page: Page) -> list[ExtractedLine]:
    """Rows carrying an HSN/SAC code, read between the table header and the totals block."""
    items: list[ExtractedLine] = []
    column = _hsn_column(page)
    tolerance = page.width * 0.08
    for line in page.lines:
        text = line.text
        if re.search(r"hsn|sac", text, re.IGNORECASE) or re.search(r"taxable\s+value|grand\s+total", text, re.IGNORECASE):
            continue
        codes = [w for w in line.words if HSN.fullmatch(w.text.strip()) and len(w.text.strip()) >= 4]
        if column is not None:
            in_column = [w for w in codes if abs(w.cx - column) <= tolerance]
            codes = in_column or ([] if codes else codes)
        money_values = MONEY.findall(text)
        if not codes and len(money_values) < 2:
            continue  # not an item row: many suppliers omit HSN, so fall back to the money columns
        anchor = codes[0] if codes else next(
            (w for w in line.words if MONEY.fullmatch(w.text.strip())), None)
        if anchor is None:
            continue  # OCR split the amounts across words, so there is no column to anchor on
        description = " ".join(w.text for w in line.words if w.x1 <= anchor.x0).strip(" :|1234567890.")
        tail = text[text.find(anchor.text) + len(anchor.text):] if codes else text
        numbers = [n for m in MONEY.findall(tail) if (n := _money(m))]
        rate = PERCENT.search(text)
        items.append(ExtractedLine(
            description=description or None,
            hsn_sac=codes[0].text.strip() if codes else None,
            unit_price=numbers[0] if numbers else None,
            taxable_value=max(numbers) if numbers else None,
            gst_rate=Decimal(rate.group(1)) if rate else None,
            region=_region(line.words, page),
        ))
    return items


def extract(page: Page, document_id: str) -> ExtractedInvoice:
    """Pull the fields out, and remember which part of the page each one came from."""
    (supplier_gstin, supplier_gstin_words), (recipient_gstin, recipient_gstin_words) = _gstins_by_role(page)
    supplier_name, supplier_name_words = _supplier_name(page)
    recipient_name, recipient_name_words = _recipient_name(page)
    invoice_number, invoice_number_words = _value(page, r"invoice\s*no\.?", r"inv\s*no\.?", r"bill\s*no\.?")
    invoice_date_text, invoice_date_words = _value(page, r"invoice\s*date", r"\bdate\b")
    due_date_text, due_date_words = _value(page, r"due\s*date")
    place_text, place_words = _value(page, r"place\s+of\s+supply")
    taxable_total, taxable_words = _amount(page, r"taxable\s+value", r"taxable\s+amount", r"\btaxable\b")
    cgst_total, cgst_words = _amount(page, r"\bcgst\b")
    sgst_total, sgst_words = _amount(page, r"\bsgst\b")
    igst_total, igst_words = _amount(page, r"\bigst\b")
    invoice_total, total_words = _amount(page, r"grand\s+total", r"amount\s+due", r"total\s+payable", r"\btotal\b")
    round_off, round_off_words = _amount(page, r"round\s*off", r"rounding")

    sources = {
        "supplier_name": supplier_name_words, "supplier_gstin": supplier_gstin_words,
        "recipient_name": recipient_name_words, "recipient_gstin": recipient_gstin_words,
        "invoice_number": invoice_number_words, "invoice_date": invoice_date_words,
        "due_date": due_date_words, "place_of_supply": place_words,
        "taxable_total": taxable_words, "cgst_total": cgst_words, "sgst_total": sgst_words,
        "igst_total": igst_words, "invoice_total": total_words, "round_off": round_off_words,
    }
    provenance = {field: region for field, words in sources.items()
                  if (region := _region(words, page)) is not None}
    mean_conf = sum(w.conf for w in page.words) / len(page.words) if page.words else 0.0

    return ExtractedInvoice(
        document_id=document_id,
        supplier_name=supplier_name,
        supplier_gstin=supplier_gstin,
        recipient_name=recipient_name,
        recipient_gstin=recipient_gstin,
        invoice_number=invoice_number,
        invoice_date=_date(invoice_date_text or ""),
        due_date=_date(due_date_text or ""),
        place_of_supply=(place_text or "").split("-")[0].strip() or None,
        taxable_total=taxable_total,
        cgst_total=cgst_total,
        sgst_total=sgst_total,
        igst_total=igst_total,
        invoice_total=invoice_total,
        round_off=round_off,
        line_items=_line_items(page),
        confidence={"ocr_mean": round(mean_conf, 4)},
        provenance=provenance,
    )
