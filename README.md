# gstpipe

Measuring how extraction errors propagate through a financial document pipeline:
**extraction → categorisation → invoice–payment reconciliation → anomaly detection**, on
GST-structured Indian invoices.

The platform is the instrument. The contribution is the measurement.

## The finding

Errors do not spread evenly. Categorisation and reconciliation **absorb** upstream error; anomaly
detection **amplifies** it, badly.

| Stage | Amplification A(s, ε) at ε = 5–30% | Behaviour |
|---|---|---|
| 2. Categorisation | 0.14 – 0.28 | absorbs |
| 3. Reconciliation | 0.38 – 0.65 | absorbs |
| 4. Anomaly detection | **2.2 – 11.1** | **amplifies** |

`A(s, ε) = [M(s, 0) − M(s, ε)] / ε` — above 1 the stage loses more accuracy than the error it was handed.

**Why anomaly detection collapses.** It relies on internal consistency: does the tax add up, is the
GSTIN valid, has this invoice number been used before. A misread digit produces exactly the same
symptom as a genuine anomaly, and the stage cannot tell them apart. As upstream error rises its
*recall goes up* (0.50 → 0.72) while *precision collapses* (1.00 → 0.04): at 30% upstream error the
rule detector flags **66% of all invoices**. The review queue stops being a queue.

That is the practical answer to where human review is load-bearing: not at the stages that quietly
cope, but at the one that turns noise into alarms.

Figures land in `data/synthetic/<name>/reports/figures/`.

## The application: Ledgerline

**Books a CA can sign off — every figure traced to the document it came from.**

Invoice tools read documents and hand you numbers. This one hands you numbers *and what it read
them from*, and refuses to present anything as settled until it can show its work. Each invoice
sits in one of three states and always says why:

| State | Meaning |
|---|---|
| **Verified** | every essential read, values located on the page, arithmetic agrees, nothing flagged — or a person signed it off |
| **Needs checking** | usable, but a check failed; the reason and the region are shown |
| **Could not read** | the essentials were not recovered, so it is excluded from every total rather than guessed at |

On real scans of this dataset that comes out as **20% verified, 51% needing a person, 29%
unreadable** — and the dashboard says so instead of averaging it away.

**This is the research finding turned into a product decision.** The experiment showed that
extraction errors become false alarms downstream: a misread digit is indistinguishable from a
genuine inconsistency, and the anomaly stage amplifies that by 2–11x. If that is true, then no
automated financial flag can be trusted on its own — so the interface never shows a number without
the pixels behind it.

`--bootstrap` takes a clean checkout to a working demo in one command.

```powershell
.\.venv\Scripts\python.exe -m gstpipe.webapp.cli --bootstrap
```

Then <http://localhost:8000>. `/upload` takes a PDF or a photo and runs the whole chain live —
OCR, field extraction, categorisation, consistency checks against that supplier's earlier
invoices — and queues anything doubtful for review. `/sample.pdf` gives you a clean invoice to try.

To share it with someone, see [DEPLOY.md](DEPLOY.md): one `cloudflared` command publishes it on a
public HTTPS URL. Cloudflare Workers and Pages cannot host the app itself (no Tesseract, no
scikit-learn), but `python -m gstpipe.export_site` builds a static results site that Pages serves
perfectly.

## Setup

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

OCR needs Tesseract (`winget install UB-Mannheim.TesseractOCR`). Everything else runs without it —
use `--source text` to read the PDF text layer instead.

## Running the whole pipeline

```powershell
$py = ".\.venv\Scripts\python.exe"
& $py -m gstpipe.generator.cli --businesses 30 --seed 2026     # synthetic data + ground truth
& $py -m gstpipe.render.cli    --dataset data/synthetic/seed-2026   # invoice images (~15 min)
& $py -m gstpipe.extract.cli   --dataset data/synthetic/seed-2026 --source text   # E1 upper bound
& $py -m gstpipe.extract.cli   --dataset data/synthetic/seed-2026 --source ocr    # E1 realistic
& $py -m gstpipe.categorise.cli --dataset data/synthetic/seed-2026                # E2
& $py -m gstpipe.reconcile.cli  --dataset data/synthetic/seed-2026                # E3
& $py -m gstpipe.anomaly.cli    --dataset data/synthetic/seed-2026                # E4
& $py -m gstpipe.inject.cli     --dataset data/synthetic/seed-2026                # E5, the study
```

Reports are written to `data/synthetic/seed-2026/reports/`.

## The five experiments

| | Question | Compared |
|---|---|---|
| **E1** | How well can fields be read? | PDF text layer vs OCR, across four layouts and scan severities |
| **E2** | Can expenses be categorised? | keyword rules vs word / character / hybrid TF-IDF models |
| **E3** | Can payments be matched to invoices? | exact vs fuzzy vs learned vs global assignment |
| **E4** | Which invoices need review? | rules vs per-supplier statistics vs Isolation Forest / LOF |
| **E5** | **How far does an extraction error travel?** | every stage, swept over injected error 0–30%, 5 seeds |

Selected results on the 30-business dataset:

- **E1** — text layer 0.999 field accuracy; OCR 0.81 → 0.35 as scan severity rises. GSTINs are the
  hardest field (0.27): fifteen characters with no redundancy. The Courier-based `compact` layout
  (0.52) is far harder than `modern` (0.77).
- **E3** — exact matching reaches F1 0.65 because it cannot handle payments net of TDS, instalments
  or consolidated transfers; a learned matcher reaches 0.96. Global assignment (0.91) scores *below*
  greedy matching, because one-to-many settlements break its one-to-one assumption.
- **E4** — rules 0.74 F1 at perfect precision, Isolation Forest 0.88 with ROC-AUC 0.99.

### Real OCR reproduces the injected result

The controlled sweep predicts what real OCR does to each stage, which is the strongest check
available on the injection model. Measured on the same test split, ground truth vs real OCR input:

| Stage | truth | real OCR | |
|---|---|---|---|
| Categorisation (hybrid) | 0.993 | 0.854 | degrades gently |
| Categorisation (keyword rules) | 0.800 | 0.631 | the rule tier suffers most |
| Reconciliation (forest) | 0.961 | 0.742 | degrades gently |
| Anomaly (Isolation Forest) | 0.884 | **0.231** | collapses |
| Anomaly (rules) | 0.743 | **0.090** | collapses |

The review application makes the same point concretely: seeded from the perfect-reader extraction it
flags 9 invoices out of 468; seeded from real OCR it flags 229.

## Layout

| Package | What it does |
|---|---|
| `gstpipe.generator` | businesses, vendors, GST invoices, bank transactions, planted anomalies, ground truth |
| `gstpipe.render` | four invoice layouts, rasterisation, scan degradation |
| `gstpipe.extract` | positioned-word page model, rule extractor, field-level scoring |
| `gstpipe.categorise` | features, keyword rules, TF-IDF model tiers |
| `gstpipe.reconcile` | pair features, four matcher tiers, precision/false-match scoring |
| `gstpipe.anomaly` | consistency and behavioural signals, three detector tiers |
| `gstpipe.inject` | controlled error injection, the propagation sweep, figures |

## What the data gets right

- GSTINs with valid checksums; PAN holder type matches legal form
- CGST+SGST intra-state, IGST inter-state; slab schedule changes on 22 Sep 2025
- Vendors bill at a consistent price level and order size, so per-supplier history means something
- Payments net of TDS, instalments, consolidated transfers, late settlement, unpaid invoices
- Bank narrations that truncate and normalise counterparty names
- 20% of suppliers omit HSN entirely; descriptions carry PO references, periods and abbreviations
- Every planted anomaly is detectable in principle, and no unlabelled invoice trips a check — both
  are enforced by tests, because a label nothing can find silently penalises every model

## Tests

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

129 tests. They cover GST arithmetic, settlement integrity, label quality, extraction on every
layout, and the alignment and detection bugs found while building this.

HSN/SAC codes and rates in `catalog.py` are illustrative for synthetic data, not a tax reference.
