"""Build a static results site.

The application needs a Python runtime and an OCR engine, so it cannot live on Cloudflare Pages.
The results can: this writes a self-contained folder of HTML and images that any static host will
serve, which is the right home for a permanent link to the findings.
"""

from __future__ import annotations

import argparse
import html
import shutil
from datetime import date
from pathlib import Path

import pandas as pd

HEADLINE = {"1_extraction": "field_accuracy", "2_categorisation": "accuracy",
            "3_reconciliation": "f1", "4_anomaly": "f1"}
STAGE_LABEL = {"1_extraction": "1 · Extraction", "2_categorisation": "2 · Categorisation",
               "3_reconciliation": "3 · Reconciliation", "4_anomaly": "4 · Anomaly detection"}
BEST = {"1_extraction": "injected", "2_categorisation": "hybrid_lr",
        "3_reconciliation": "m3_forest", "4_anomaly": "a3_isolation_forest"}

CSS = """
:root{--navy:#1a2a4a;--accent:#2f6fed;--grey:#5b6472;--line:#dde3ee;--bg:#f6f8fc;--bad:#c33;--good:#2e7d4f}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:#1d2430;font:16px/1.65 "Segoe UI",system-ui,sans-serif}
header{background:var(--navy);color:#fff;padding:40px 24px 34px}
header div{max-width:960px;margin:0 auto}
h1{margin:0 0 8px;font-size:30px;line-height:1.25}
header p{margin:0;color:#c9d6f0;font-size:17px;max-width:640px}
main{max-width:960px;margin:0 auto;padding:30px 24px 70px}
h2{font-size:21px;color:var(--navy);margin:38px 0 10px}
h3{font-size:15px;color:var(--grey);margin:24px 0 8px;text-transform:uppercase;letter-spacing:.05em}
p{margin:0 0 14px}
.lead{font-size:17px}
.panel{background:#fff;border:1px solid var(--line);border-radius:12px;padding:20px 22px;margin:16px 0}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:14px;margin:18px 0}
.card{background:#fff;border:1px solid var(--line);border-radius:12px;padding:16px 18px}
.card .n{font-size:26px;font-weight:700;color:var(--navy)}
.card .n.bad{color:var(--bad)} .card .n.good{color:var(--good)}
.card .l{font-size:12px;color:var(--grey);text-transform:uppercase;letter-spacing:.05em}
table{width:100%;border-collapse:collapse;background:#fff;font-size:14px;border:1px solid var(--line);border-radius:10px;overflow:hidden}
th,td{text-align:left;padding:9px 12px;border-bottom:1px solid var(--line)}
th{font-size:12px;text-transform:uppercase;letter-spacing:.04em;color:var(--grey);background:#f3f6fc}
td.num,th.num{text-align:right;font-variant-numeric:tabular-nums}
tr:last-child td{border-bottom:0}
figure{margin:18px 0}
figure img{width:100%;border:1px solid var(--line);border-radius:10px;background:#fff}
figcaption{font-size:13px;color:var(--grey);margin-top:8px}
.strip{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px}
.strip img{width:100%;border:1px solid var(--line);border-radius:8px;background:#fff}
.strip span{display:block;font-size:12px;color:var(--grey);margin-top:5px}
code{background:#eef2fa;padding:2px 6px;border-radius:5px;font-size:13px}
footer{border-top:1px solid var(--line);margin-top:44px;padding-top:18px;color:var(--grey);font-size:13px}
a{color:var(--accent)}
"""


def _table(frame: pd.DataFrame, numeric_cols: tuple[str, ...] = ()) -> str:
    head = "".join(f'<th class="{"num" if c in numeric_cols else ""}">{html.escape(str(c))}</th>'
                   for c in frame.columns)
    rows = []
    for _, row in frame.iterrows():
        cells = "".join(
            f'<td class="{"num" if c in numeric_cols else ""}">{html.escape(str(row[c]))}</td>'
            for c in frame.columns)
        rows.append(f"<tr>{cells}</tr>")
    return f"<table><tr>{head}</tr>{''.join(rows)}</table>"


def _read(path: Path) -> pd.DataFrame | None:
    return pd.read_csv(path) if path.exists() else None


def build(dataset: Path, out: Path) -> Path:
    reports = dataset / "reports"
    out.mkdir(parents=True, exist_ok=True)
    assets = out / "assets"
    assets.mkdir(exist_ok=True)

    figures = []
    for name in ("degradation_curves.png", "amplification.png", "anomaly_mechanism.png"):
        source = reports / "figures" / name
        if source.exists():
            shutil.copy(source, assets / name)
            figures.append(name)

    samples = []
    docs = sorted((dataset / "documents").glob("*.jpg"))
    index = _read(reports.parent / "ground_truth" / "documents.csv")
    if docs and index is not None:
        picks = (index.sort_values("severity").iloc[[0, len(index) // 2, -1]]
                 if len(index) >= 3 else index)
        for _, row in picks.iterrows():
            source = dataset / row.file
            if source.exists():
                shutil.copy(source, assets / source.name)
                samples.append((source.name, row.template, float(row.severity)))

    summary = _read(reports / "propagation_summary.csv")
    parts: list[str] = []

    if summary is not None:
        mask = summary.apply(lambda r: HEADLINE.get(r.stage) == r.metric, axis=1)
        headline = summary[mask]
        amp = headline[(headline.rate > 0)]
        cards = []
        for stage, label in STAGE_LABEL.items():
            rows = amp[(amp.stage == stage) & (amp.method == BEST[stage])]
            if rows.empty or stage == "1_extraction":
                continue
            lo, hi = rows.amplification.min(), rows.amplification.max()
            tone = "bad" if hi > 1 else "good"
            verdict = "amplifies" if hi > 1 else "absorbs"
            cards.append(f'<div class="card"><div class="l">{html.escape(label)}</div>'
                         f'<div class="n {tone}">{lo:.2f} – {hi:.2f}</div>'
                         f'<div class="l">{verdict}</div></div>')
        parts.append(f'<div class="cards">{"".join(cards)}</div>')

        curve = headline.pivot_table(index=["stage", "method"], columns="rate",
                                     values="mean").round(3).reset_index()
        curve.columns = [c if isinstance(c, str) else f"{float(c):.0%}" for c in curve.columns]
        parts.append("<h3>Headline metric by injected error rate</h3>")
        parts.append(_table(curve, tuple(c for c in curve.columns if c.endswith("%"))))

    for title, path, note in (
        ("Extraction (E1)", reports / "extraction_ocr_by_severity.csv",
         "Field accuracy against how badly the scan was degraded."),
        ("Categorisation (E2)", reports / "categorisation.csv",
         "Keyword rules against text models, on clean fields and on real OCR output."),
        ("Reconciliation (E3)", reports / "reconciliation.csv",
         "Exact, fuzzy, learned and globally assigned matching."),
        ("Anomaly detection (E4)", reports / "anomaly.csv",
         "Rules, per-supplier statistics and unsupervised models."),
    ):
        frame = _read(path)
        if frame is None:
            continue
        if "condition" in frame.columns:
            value = "f1" if "f1" in frame.columns else "accuracy"
            index = [c for c in ("model", "matcher", "detector") if c in frame.columns][0]
            frame = frame.pivot(index=index, columns="condition", values=value).round(3).reset_index()
        parts.append(f"<h3>{html.escape(title)}</h3><p>{html.escape(note)}</p>")
        parts.append(_table(frame.round(3) if hasattr(frame, "round") else frame,
                            tuple(c for c in frame.columns if c not in
                                  ("model", "matcher", "detector", "template", "severity_bucket"))))

    figure_html = "".join(
        f'<figure><img src="assets/{name}" alt="{name}"><figcaption>'
        f'{html.escape(name.replace("_", " ").replace(".png", ""))}</figcaption></figure>'
        for name in figures)

    strip = "".join(
        f'<div><img src="assets/{name}" alt="rendered invoice">'
        f'<span>{html.escape(template)} · severity {severity:.2f}</span></div>'
        for name, template, severity in samples)

    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Error propagation in a financial document pipeline</title>
<style>{CSS}</style></head>
<body>
<header><div>
<h1>How far does an extraction error travel?</h1>
<p>A four-stage pipeline over GST invoices, measured stage by stage as upstream errors are injected
at a controlled rate.</p>
</div></header>
<main>
<p class="lead">Existing work evaluates each stage on clean input. In production every stage reads
whatever the one before it produced. This measures the difference.</p>

<div class="panel"><strong>The finding.</strong> Errors do not spread evenly. Categorisation and
reconciliation absorb upstream error. Anomaly detection amplifies it: its recall rises while
precision collapses, because a misread digit looks exactly like a genuine inconsistency. At 30%
upstream error the rule detector flags two thirds of all invoices, and the review queue stops being
useful.</div>

{''.join(parts)}

<h2>Figures</h2>
{figure_html or "<p>No figures found. Run the experiment first.</p>"}

<h2>The documents</h2>
<p>Every invoice is rendered in one of four layouts, then degraded to look scanned or photographed.</p>
<div class="strip">{strip}</div>

<h2>Reproducing this</h2>
<div class="panel">
<p>Everything here is generated from a seed, so the dataset is never committed.</p>
<p><code>python -m gstpipe.generator.cli --businesses 30 --seed 2026</code><br>
<code>python -m gstpipe.render.cli --dataset data/synthetic/seed-2026</code><br>
<code>python -m gstpipe.inject.cli --dataset data/synthetic/seed-2026</code></p>
<p>The review application needs a Python runtime and Tesseract, so it is run separately — see
<code>DEPLOY.md</code>.</p>
</div>

<footer>Generated {date.today().isoformat()} · all data is synthetic · HSN/SAC codes and rates are
illustrative, not a tax reference.</footer>
</main></body></html>
"""
    (out / "index.html").write_text(page, encoding="utf-8")
    (out / "_headers").write_text("/*\n  X-Frame-Options: DENY\n  X-Content-Type-Options: nosniff\n",
                                  encoding="utf-8")
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the static results site.")
    parser.add_argument("--dataset", type=Path, default=Path("data/synthetic/seed-2026"))
    parser.add_argument("--out", type=Path, default=Path("site"))
    args = parser.parse_args(argv)

    out = build(args.dataset, args.out)
    files = sorted(p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file())
    print(f"wrote {out} ({len(files)} files)")
    for name in files[:12]:
        print(f"  {name}")
    print("\nPublish it with:  npx wrangler pages deploy site --project-name gstpipe")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
