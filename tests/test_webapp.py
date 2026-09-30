import importlib
from datetime import date

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from gstpipe.generator.dataset import DatasetConfig, build_dataset, write_dataset
from gstpipe.inject.corrupt import from_invoice
from gstpipe.webapp import db
from gstpipe.webapp.seed import seed_database


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    out = tmp_path_factory.mktemp("web")
    config = DatasetConfig(seed=5, n_businesses=4, start=date(2025, 10, 1), end=date(2026, 1, 31))
    ds = build_dataset(config)
    write_dataset(ds, out)
    # A perfect extraction stands in for Stage 1 so the app has values without running OCR.
    extractions = out / "extractions"
    extractions.mkdir(exist_ok=True)
    with (extractions / "text.jsonl").open("w", encoding="utf-8") as fh:
        for invoice in ds.invoices:
            fh.write(from_invoice(invoice).model_dump_json() + "\n")
    return out


@pytest.fixture(scope="module")
def client(dataset, tmp_path_factory, monkeypatch_module=None):
    import os
    db_path = tmp_path_factory.mktemp("db") / "review.db"
    counts = seed_database(dataset, db_path, businesses=2, progress=lambda *a: None)
    assert counts["invoices"] > 0
    os.environ["GSTPIPE_DATASET"] = str(dataset)
    os.environ["GSTPIPE_DB"] = str(db_path)
    from gstpipe.webapp import app as app_module
    importlib.reload(app_module)
    return TestClient(app_module.app), db_path


def test_seeding_loads_only_the_requested_businesses(client):
    _, db_path = client
    engine = db.make_engine(db_path)
    with Session(engine) as session:
        assert len(session.scalars(db.select(db.Business)).all()) == 2
        assert session.scalars(db.select(db.Invoice)).first() is not None


def test_pages_render(client):
    http, _ = client
    for path in ("/", "/review", "/payments", "/analytics", "/documents"):
        response = http.get(path)
        assert response.status_code == 200, path
        assert "Ledgerline" in response.text, path


def test_invoice_page_shows_model_values(client):
    http, db_path = client
    engine = db.make_engine(db_path)
    with Session(engine) as session:
        invoice = session.scalars(db.select(db.Invoice)).first()
        invoice_id, number = invoice.id, invoice.invoice_number
    response = http.get(f"/invoices/{invoice_id}")
    assert response.status_code == 200
    assert number in response.text
    assert "History" in response.text


def test_correcting_a_field_is_recorded_against_the_model_value(client):
    http, db_path = client
    engine = db.make_engine(db_path)
    with Session(engine) as session:
        invoice = session.scalars(db.select(db.Invoice)).first()
        invoice_id = invoice.id
        original = {f.field: f.approved_value for f in invoice.fields}

    payload = dict(original)
    payload["supplier_name"] = "Corrected Supplier Pvt Ltd"
    response = http.post(f"/invoices/{invoice_id}/save", data=payload, follow_redirects=False)
    assert response.status_code == 303

    with Session(engine) as session:
        invoice = session.get(db.Invoice, invoice_id)
        field = next(f for f in invoice.fields if f.field == "supplier_name")
        assert field.approved_value == "Corrected Supplier Pvt Ltd"
        assert field.corrected is True
        assert field.model_value == original["supplier_name"]  # the original reading is kept
        events = session.scalars(db.select(db.AuditEvent)
                                 .where(db.AuditEvent.invoice_id == invoice_id)).all()
        assert any(e.field == "supplier_name" and e.new_value == "Corrected Supplier Pvt Ltd"
                   for e in events)


def test_approving_clears_flags_and_records_the_action(client):
    http, db_path = client
    engine = db.make_engine(db_path)
    with Session(engine) as session:
        flagged = session.scalars(db.select(db.Invoice)
                                  .where(db.Invoice.status == "needs_review")).first()
        if flagged is None:
            pytest.skip("nothing flagged in this sample")
        invoice_id = flagged.id

    assert http.post(f"/invoices/{invoice_id}/approve", follow_redirects=False).status_code == 303
    with Session(engine) as session:
        invoice = session.get(db.Invoice, invoice_id)
        assert invoice.status == "approved"
        assert all(f.resolved for f in invoice.flags)
        assert any(e.action == "approve" for e in session.scalars(
            db.select(db.AuditEvent).where(db.AuditEvent.invoice_id == invoice_id)).all())


def test_json_api_mirrors_the_pages(client):
    http, _ = client
    stats = http.get("/api/stats")
    assert stats.status_code == 200 and "invoice_count" in stats.json()
    queue = http.get("/api/queue")
    assert queue.status_code == 200 and isinstance(queue.json(), list)


def test_unknown_invoice_is_a_404(client):
    http, _ = client
    assert http.get("/invoices/does-not-exist").status_code == 404


# ----------------------------------------------------------------- live processing

@pytest.fixture(scope="module")
def live(dataset, tmp_path_factory):
    """The app with a trained pipeline, for the upload path."""
    import os

    from gstpipe.artifacts import save, train

    model_path = tmp_path_factory.mktemp("model") / "pipeline.joblib"
    save(train(dataset, progress=lambda *a: None), model_path)
    db_path = tmp_path_factory.mktemp("livedb") / "review.db"
    seed_database(dataset, db_path, businesses=2, progress=lambda *a: None)

    os.environ["GSTPIPE_DATASET"] = str(dataset)
    os.environ["GSTPIPE_DB"] = str(db_path)
    os.environ["GSTPIPE_MODEL"] = str(model_path)
    os.environ["GSTPIPE_UPLOADS"] = str(tmp_path_factory.mktemp("uploads"))
    from gstpipe.webapp import app as app_module
    importlib.reload(app_module)
    return TestClient(app_module.app), db_path, dataset


def test_health_reports_the_loaded_model(live):
    http, _, _ = live
    body = http.get("/api/health").json()
    assert body["model"] and body["model"]["train_invoices"] > 0
    assert body["model_error"] is None


def test_sample_invoice_downloads_as_a_pdf(live):
    http, _, _ = live
    response = http.get("/sample.pdf")
    assert response.status_code == 200
    assert response.content[:5] == b"%PDF-"


def test_upload_rejects_an_empty_file(live):
    http, db_path, _ = live
    with Session(db.make_engine(db_path)) as session:
        business_id = session.scalars(db.select(db.Business)).first().id
    response = http.post("/upload", files={"file": ("empty.pdf", b"", "application/pdf")},
                         data={"business_id": business_id})
    assert response.status_code == 400


def test_upload_rejects_an_unknown_business(live):
    http, _, _ = live
    response = http.post("/upload", files={"file": ("x.pdf", b"%PDF-1.4 broken", "application/pdf")},
                         data={"business_id": "nope"})
    assert response.status_code == 400


@pytest.mark.skipif(not __import__("gstpipe.extract.sources", fromlist=["x"]).tesseract_available(),
                    reason="tesseract is not installed")
def test_uploading_a_clean_invoice_reads_it_back(live):
    from gstpipe.render.pipeline import to_pdf
    from gstpipe.schema import Invoice

    http, db_path, dataset = live
    source = (dataset / "ground_truth" / "invoices.jsonl").read_text("utf-8").splitlines()
    invoice = next(Invoice.model_validate_json(line) for line in source
                   if len(Invoice.model_validate_json(line).line_items) >= 1)
    pdf = to_pdf(invoice, "classic")

    with Session(db.make_engine(db_path)) as session:
        business_id = session.scalars(db.select(db.Business)).first().id

    response = http.post("/upload", files={"file": ("invoice.pdf", pdf, "application/pdf")},
                         data={"business_id": business_id}, follow_redirects=False)
    assert response.status_code == 303
    invoice_id = response.headers["location"].rsplit("/", 1)[-1]

    body = http.get(f"/api/invoices/{invoice_id}").json()
    read = {field: values["model"] for field, values in body["fields"].items()}
    assert read["invoice_number"] == invoice.invoice_number
    assert read["invoice_total"] == str(invoice.invoice_total)
    # the original reading is what was stored, and nothing is marked corrected yet
    assert all(not values["corrected"] for values in body["fields"].values())
    assert any(event["action"] == "upload" for event in body["audit"])


@pytest.mark.skipif(not __import__("gstpipe.extract.sources", fromlist=["x"]).tesseract_available(),
                    reason="tesseract is not installed")
def test_api_process_returns_fields_without_storing_anything(live):
    from gstpipe.render.pipeline import to_pdf
    from gstpipe.schema import Invoice

    http, db_path, dataset = live
    line = (dataset / "ground_truth" / "invoices.jsonl").read_text("utf-8").splitlines()[0]
    invoice = Invoice.model_validate_json(line)

    with Session(db.make_engine(db_path)) as session:
        before = len(session.scalars(db.select(db.Invoice)).all())

    body = http.post("/api/process",
                     files={"file": ("x.pdf", to_pdf(invoice, "modern"), "application/pdf")}).json()
    assert body["fields"]["invoice_number"] == invoice.invoice_number
    assert 0.0 <= body["ocr_confidence"] <= 1.0
    assert isinstance(body["flags"], list)

    with Session(db.make_engine(db_path)) as session:
        assert len(session.scalars(db.select(db.Invoice)).all()) == before


@pytest.mark.skipif(not __import__("gstpipe.extract.sources", fromlist=["x"]).tesseract_available(),
                    reason="tesseract is not installed")
def test_uploaded_document_image_is_reachable(live):
    """The invoice page must link the stored file, which lives under a different prefix to the dataset."""
    import re

    from gstpipe.render.pipeline import to_pdf
    from gstpipe.schema import Invoice

    http, db_path, dataset = live
    line = (dataset / "ground_truth" / "invoices.jsonl").read_text("utf-8").splitlines()[1]
    invoice = Invoice.model_validate_json(line)
    with Session(db.make_engine(db_path)) as session:
        business_id = session.scalars(db.select(db.Business)).first().id

    response = http.post("/upload", files={"file": ("x.pdf", to_pdf(invoice, "letterhead"),
                                                    "application/pdf")},
                         data={"business_id": business_id}, follow_redirects=False)
    page = http.get(response.headers["location"]).text
    match = re.search(r'<img src="([^"]+)" alt="the scanned invoice this was read from"', page)
    assert match, "the invoice page did not render its source document"
    src = match.group(1)
    assert src.startswith("/uploads/"), src
    assert http.get(src).status_code == 200


# ----------------------------------------------------------------- provenance and analytics

def test_analytics_only_counts_invoices_it_could_read(client):
    """The figures must state their own denominator rather than quietly dropping documents."""
    from gstpipe.webapp.analytics import build

    _, db_path = client
    with Session(db.make_engine(db_path)) as session:
        data = build(session)
    coverage = data["coverage"]
    assert coverage["invoices"] >= coverage["usable"]
    assert coverage["usable"] == data["invoice_count"]
    assert coverage["invoices"] - coverage["usable"] == (
        coverage["no_date_or_total"] + coverage["implausible_date"])


def test_a_misread_year_cannot_drag_the_charts(client):
    """One invoice dated decades out would otherwise stretch every month axis."""
    from datetime import date as _date

    from gstpipe.webapp.analytics import build

    _, db_path = client
    engine = db.make_engine(db_path)
    with Session(engine) as session:
        invoice = session.scalars(db.select(db.Invoice)
                                  .where(db.Invoice.invoice_date.is_not(None))).first()
        invoice.invoice_date = _date(2099, 1, 1)
        session.commit()
    with Session(engine) as session:
        data = build(session)
    assert data["as_of"].year < 2099
    assert data["coverage"]["implausible_date"] >= 1


@pytest.mark.skipif(not __import__("gstpipe.extract.sources", fromlist=["x"]).tesseract_available(),
                    reason="tesseract is not installed")
def test_uploaded_invoice_records_where_each_value_was_read(live):
    """Provenance is the point: a value must be traceable to a region of the page."""
    import json as _json

    from gstpipe.render.pipeline import to_pdf
    from gstpipe.schema import Invoice

    http, db_path, dataset = live
    line = (dataset / "ground_truth" / "invoices.jsonl").read_text("utf-8").splitlines()[2]
    invoice = Invoice.model_validate_json(line)
    with Session(db.make_engine(db_path)) as session:
        business_id = session.scalars(db.select(db.Business)).first().id

    response = http.post("/upload", files={"file": ("x.pdf", to_pdf(invoice, "classic"),
                                                    "application/pdf")},
                         data={"business_id": business_id}, follow_redirects=False)
    invoice_id = response.headers["location"].rsplit("/", 1)[-1]

    with Session(db.make_engine(db_path)) as session:
        stored = session.get(db.Invoice, invoice_id)
        traced = [f for f in stored.fields if f.region]
        assert traced, "no field recorded where it came from"
        for value in traced:
            region = _json.loads(value.region)
            assert region["x1"] > region["x0"] and region["y1"] > region["y0"]
            assert 0 < region["page_width"] and 0 < region["page_height"]

    page = http.get(f"/invoices/{invoice_id}").text
    assert 'class="hit"' in page, "the page did not draw the traced regions"
