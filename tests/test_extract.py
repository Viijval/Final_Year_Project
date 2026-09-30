import random
from datetime import date
from decimal import Decimal

import pytest

from gstpipe.extract.evaluate import Outcome, compare, compare_field, normalise, score
from gstpipe.extract.page import Page, Word
from gstpipe.extract.records import FIELDS
from gstpipe.extract.rules import extract
from gstpipe.extract.sources import from_pdf
from gstpipe.generator.entities import make_business
from gstpipe.generator.invoices import make_invoices
from gstpipe.render.pipeline import TEMPLATE_NAMES, to_pdf

START, END = date(2025, 4, 1), date(2026, 3, 31)


@pytest.fixture(scope="module")
def invoices():
    rng = random.Random(31)
    return make_invoices(rng, make_business(rng, "B000", "train", set()), START, END)


def _extract_clean(invoice, template):
    return extract(from_pdf(to_pdf(invoice, template)), invoice.invoice_id)


# ----------------------------------------------------------------- page model

def test_words_group_into_lines_by_vertical_position():
    words = [Word("Invoice", 10, 10, 60, 20), Word("No", 64, 11, 80, 21), Word("Total", 10, 60, 50, 70)]
    page = Page(words, width=595, height=842)
    assert [line.text for line in page.lines] == ["Invoice No", "Total"]


def test_find_matches_case_insensitively_in_reading_order():
    page = Page([Word("GSTIN", 10, 10, 50, 20), Word("gstin", 10, 60, 50, 70)], 595, 842)
    assert len(page.find(r"gstin")) == 2


def test_scale_to_rescales_coordinates():
    page = Page([Word("x", 10, 20, 30, 40)], 100, 200).scale_to(200, 400)
    assert (page.words[0].x0, page.words[0].y0) == (20, 40)


# ----------------------------------------------------------------- scoring

def test_text_comparison_ignores_case_and_punctuation():
    assert compare_field("supplier_name", "Iyer & Rao Law LLP", "IYER  RAO LAW LLP") is Outcome.correct


def test_money_comparison_allows_one_paisa():
    assert compare_field("invoice_total", Decimal("100.00"), Decimal("100.01")) is Outcome.correct
    assert compare_field("invoice_total", Decimal("100.00"), Decimal("100.50")) is Outcome.wrong


def test_missing_is_distinct_from_wrong():
    assert compare_field("invoice_number", "INV-1", None) is Outcome.missing
    assert compare_field("invoice_number", "INV-1", "INV-2") is Outcome.wrong


def test_normalise_strips_gstin_spacing():
    assert normalise("supplier_gstin", "27 aapfu0939f1zv") == "27AAPFU0939F1ZV"


def test_score_counts_outcomes_and_separates_precision_from_accuracy():
    rows = [{"invoice_total": Outcome.correct, "due_date": Outcome.missing},
            {"invoice_total": Outcome.wrong, "due_date": Outcome.correct}]
    overall, per_field = score(rows)
    assert (overall.correct, overall.wrong, overall.missing) == (2, 1, 1)
    assert per_field["invoice_total"].accuracy == 0.5
    assert per_field["due_date"].precision == 1.0  # it only committed once, and was right


# ----------------------------------------------------------------- rule extractor on a perfect reader

@pytest.mark.parametrize("template", TEMPLATE_NAMES)
def test_headline_fields_recovered_from_the_text_layer(invoices, template):
    invoice = next(i for i in invoices if len(i.line_items) > 1)
    outcomes = compare(invoice, _extract_clean(invoice, template))
    for field in ("supplier_gstin", "recipient_gstin", "invoice_number", "invoice_date",
                  "invoice_total", "taxable_total", "place_of_supply"):
        assert outcomes[field] is Outcome.correct, f"{template}: {field} -> {outcomes[field]}"


@pytest.mark.parametrize("template", TEMPLATE_NAMES)
def test_party_names_recovered_without_bleeding_neighbouring_columns(invoices, template):
    invoice = invoices[0]
    outcomes = compare(invoice, _extract_clean(invoice, template))
    assert outcomes["supplier_name"] is Outcome.correct, template
    assert outcomes["recipient_name"] is Outcome.correct, template


def test_line_items_are_found_with_their_hsn_codes(invoices):
    """Some suppliers omit HSN entirely, so only printed codes are expected back."""
    invoice = next(i for i in invoices if len(i.line_items) > 1 and all(li.hsn_sac for li in i.line_items))
    extracted = _extract_clean(invoice, "classic")
    assert len(extracted.line_items) == len(invoice.line_items)
    assert {li.hsn_sac for li in extracted.line_items} == {li.hsn_sac for li in invoice.line_items}


def test_item_rows_are_still_found_when_hsn_is_omitted(invoices):
    invoice = next((i for i in invoices if any(not li.hsn_sac for li in i.line_items)), None)
    if invoice is None:
        pytest.skip("no invoice without HSN in this sample")
    extracted = _extract_clean(invoice, "classic")
    assert len(extracted.line_items) == len(invoice.line_items)


def test_tax_split_follows_the_document(invoices):
    intra = next(i for i in invoices if i.supplier.state_code == i.place_of_supply)
    extracted = _extract_clean(intra, "classic")
    assert extracted.cgst_total == intra.cgst_total and extracted.sgst_total == intra.sgst_total


def test_extraction_reports_every_field_key(invoices):
    outcomes = compare(invoices[0], _extract_clean(invoices[0], "modern"))
    assert set(outcomes) == set(FIELDS)


# ----------------------------------------------------------------- downstream robustness

@pytest.mark.parametrize("amount", [Decimal("-1250.00"), Decimal("-0.45"), Decimal("0"), None])
def test_category_features_survive_implausible_extracted_amounts(amount):
    """OCR can return a negative or missing total; features must not raise on it."""
    from gstpipe.categorise.features import CategoryInput

    numeric = CategoryInput(invoice_id="x", supplier_name="Acme", amount=amount).numeric
    assert numeric["log_amount"] >= 0.0
    assert all(isinstance(v, float) for v in numeric.values())
