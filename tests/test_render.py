import random
from datetime import date

import pymupdf
import pytest
from PIL import Image

from gstpipe.generator.entities import make_business
from gstpipe.generator.invoices import make_invoices
from gstpipe.render.layout import inr
from gstpipe.render.pipeline import (TEMPLATE_NAMES, render_invoice, render_to_file, rasterise,
                                     template_for_supplier, to_pdf)
from gstpipe.render.templates import TEMPLATES

START, END = date(2025, 4, 1), date(2026, 3, 31)


@pytest.fixture(scope="module")
def invoices():
    rng = random.Random(17)
    business = make_business(rng, "B000", "train", set())
    return make_invoices(rng, business, START, END)


@pytest.fixture(scope="module")
def sample(invoices):
    return next(i for i in invoices if len(i.line_items) > 1)


def _pdf_text(invoice, template):
    with pymupdf.open(stream=to_pdf(invoice, template), filetype="pdf") as doc:
        return doc[0].get_text()


@pytest.mark.parametrize("template", TEMPLATE_NAMES)
def test_every_template_prints_the_key_fields(sample, template):
    text = " ".join(_pdf_text(sample, template).split())
    # Templates may print party names in capitals, as real invoices do, so names compare case-insensitively.
    folded = text.upper()
    assert sample.invoice_number in text
    assert sample.supplier.gstin in text and sample.recipient.gstin in text
    assert sample.supplier.name.upper() in folded and sample.recipient.name.upper() in folded
    assert inr(sample.invoice_total) in text
    for line in sample.line_items:
        assert line.description in text and line.hsn_sac in text


@pytest.mark.parametrize("template", TEMPLATE_NAMES)
def test_every_template_fits_on_one_page(sample, template):
    with pymupdf.open(stream=to_pdf(sample, template), filetype="pdf") as doc:
        assert doc.page_count == 1


def test_intra_state_shows_cgst_sgst_and_inter_state_shows_igst(invoices):
    intra = next(i for i in invoices if i.supplier.state_code == i.place_of_supply)
    inter = next((i for i in invoices if i.supplier.state_code != i.place_of_supply), None)
    assert "CGST" in _pdf_text(intra, "classic") and "IGST" not in _pdf_text(intra, "classic")
    if inter:
        assert "IGST" in _pdf_text(inter, "classic")


def test_rasterise_gives_an_a4_page_at_the_requested_dpi(sample):
    image = rasterise(to_pdf(sample, "classic"), dpi=150)
    assert isinstance(image, Image.Image)
    assert 1200 < image.width < 1280 and 1700 < image.height < 1800


def test_supplier_keeps_one_template_and_the_set_is_used(invoices):
    supplier = invoices[0].supplier.party_id
    assert len({template_for_supplier(supplier) for _ in range(5)}) == 1
    assert {template_for_supplier(f"V{i:04d}") for i in range(200)} == set(TEMPLATES)


def test_rendering_is_reproducible(sample):
    a, name_a, spec_a = render_invoice(sample, dpi=72, severity=(0.4, 0.4), seed=3)
    b, name_b, spec_b = render_invoice(sample, dpi=72, severity=(0.4, 0.4), seed=3)
    assert name_a == name_b and spec_a == spec_b
    assert a.tobytes() == b.tobytes()


def test_higher_severity_damages_the_image_more(sample):
    clean = rasterise(to_pdf(sample, "classic"), dpi=72).convert("L")
    light, _, light_spec = render_invoice(sample, dpi=72, severity=(0.0, 0.0), seed=5)
    heavy, _, heavy_spec = render_invoice(sample, dpi=72, severity=(1.0, 1.0), seed=5)

    def distance(img):
        a, b = img.convert("L").resize(clean.size), clean
        return sum(abs(x - y) for x, y in zip(a.tobytes(), b.tobytes())) / len(b.tobytes())

    assert distance(heavy) > distance(light)
    assert heavy_spec.blur_radius > light_spec.blur_radius
    assert heavy_spec.jpeg_quality < light_spec.jpeg_quality


def test_render_to_file_writes_an_image_and_an_index_row(sample, tmp_path):
    doc = render_to_file(sample, tmp_path, dpi=100, severity=(0.3, 0.3), seed=11)
    written = tmp_path / f"{sample.invoice_id}.jpg"
    assert written.exists() and written.stat().st_size > 5_000
    row = doc.as_row()
    assert row["invoice_id"] == sample.invoice_id and row["template"] in TEMPLATE_NAMES
    assert row["file"].endswith(".jpg") and 0 <= row["severity"] <= 1
