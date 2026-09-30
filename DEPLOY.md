# Running and sharing the app

## What Cloudflare can and cannot do here

Cloudflare **Workers and Pages cannot host this application**. They run JavaScript (and a limited
Pyodide Python) at the edge with no ability to install Tesseract, scikit-learn, PyMuPDF or NumPy.
There is no configuration that makes it work — the runtime is the wrong shape for the stack.

What does work:

| Route | What it is | Good for |
|---|---|---|
| **Cloudflare Tunnel** | Cloudflare gives a public `https://…trycloudflare.com` URL that forwards to the app running on your machine or a server | Sharing with teammates today, demos, the viva |
| **Cloudflare Pages** | A static export: the figures, reports and numbers, no live processing | A permanent link that is always up |
| Any container host | Render, Fly.io, Railway, a college VM — then put Cloudflare in front for DNS and TLS | Keeping it online after the project |

## Share it in one minute (Cloudflare Tunnel)

No Cloudflare account, no DNS, no config file.

```powershell
# 1. start the app
cd pipeline
.\.venv\Scripts\python.exe -m gstpipe.webapp.cli --host 127.0.0.1 --port 8000

# 2. in a second terminal, publish it
winget install --id Cloudflare.cloudflared
cloudflared tunnel --url http://localhost:8000
```

`cloudflared` prints a URL like `https://spare-brick-quiet-mix.trycloudflare.com`. Send that to your
teammates. It stays alive while both commands are running.

A quick tunnel is anonymous and temporary. For a stable address, use a named tunnel on a domain you
control: `cloudflared tunnel login`, `cloudflared tunnel create gstpipe`, then route a hostname to it.

## Docker

The image installs Tesseract and builds the demo data during the build, so the container starts
ready to use.

```powershell
cd pipeline
docker compose up --build
```

That runs the app on <http://localhost:8000> **and** a `cloudflared` sidecar. The public URL appears
in the tunnel logs:

```powershell
docker compose logs tunnel | Select-String trycloudflare
```

Uploads and corrections live in a named volume, so they survive restarts.

## From a clean checkout, without Docker

```powershell
cd pipeline
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
winget install UB-Mannheim.TesseractOCR      # needed to read uploads

.\.venv\Scripts\python.exe -m gstpipe.webapp.cli --bootstrap
```

`--bootstrap` generates the data, renders the documents, trains the pipeline, seeds the database and
then serves. It skips any step whose output already exists, so it is safe to re-run. Rendering is
the slow part; `--businesses 3` keeps it to a couple of minutes.

## Checking it is healthy

```powershell
curl http://localhost:8000/api/health
```

```json
{"status": "ok", "model": {"train_invoices": 2910, ...}, "ocr": true}
```

`"status": "degraded"` means either Tesseract is missing or no trained model was found — the page at
`/upload` says which.

## Environment variables

| Variable | Default | Meaning |
|---|---|---|
| `GSTPIPE_DATASET` | `data/synthetic/seed-2026` | dataset the documents and samples come from |
| `GSTPIPE_DB` | `data/review.db` | SQLite database |
| `GSTPIPE_MODEL` | `models/pipeline.joblib` | trained pipeline |
| `GSTPIPE_UPLOADS` | `data/uploads` | where uploaded documents are kept |
| `HOST` / `PORT` | `127.0.0.1` / `8000` | bind address |

## Pushing to GitHub

The repository is already initialised and committed locally. Create an empty repository on GitHub
(no README, no licence), then:

```powershell
cd E:\V\Study_Work\projects\final_year
git remote add origin https://github.com/<you>/<repo>.git
git push -u origin main
```

Generated data, models and the database are ignored, so the push is small. Teammates then run
`--bootstrap` to recreate everything locally.

## A note for the demo

Uploading a heavily degraded scan will return very little — that is the project's finding, not a
fault. `/sample.pdf` gives a clean invoice to demonstrate the working path; the degraded documents
in `data/synthetic/*/documents/` show the failure path.
