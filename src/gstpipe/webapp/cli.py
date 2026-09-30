"""Set up and run the review application.

``--bootstrap`` takes a clean checkout to a working demo: generate data, render documents, train
the pipeline, seed the database. ``--seed`` alone re-seeds from a dataset that already exists.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def bootstrap(dataset: Path, db_path: Path, model_path: Path, businesses: int,
              dpi: int, progress=print) -> None:
    from gstpipe.artifacts import save, train
    from gstpipe.extract.cli import main as extract_main
    from gstpipe.generator.cli import main as generate_main
    from gstpipe.render.cli import main as render_main
    from gstpipe.webapp.seed import seed_database

    if not (dataset / "ground_truth" / "invoices.jsonl").exists():
        progress("1/5 generating synthetic data")
        generate_main(["--out", str(dataset), "--businesses", str(businesses), "--force"])
    else:
        progress("1/5 dataset already present, keeping it")

    if not (dataset / "ground_truth" / "documents.csv").exists():
        progress("2/5 rendering invoice documents (a few minutes)")
        render_main(["--dataset", str(dataset), "--dpi", str(dpi)])
    else:
        progress("2/5 documents already rendered")

    if not (dataset / "extractions" / "text.jsonl").exists():
        progress("3/5 running extraction")
        extract_main(["--dataset", str(dataset), "--source", "text"])
    else:
        progress("3/5 extraction already present")

    if not model_path.exists():
        progress("4/5 training the pipeline")
        save(train(dataset, progress=lambda m: progress("    " + m)), model_path)
    else:
        progress("4/5 trained pipeline already present")

    progress("5/5 seeding the review database")
    counts = seed_database(dataset, db_path, businesses or None, progress=lambda m: progress("    " + m))
    progress("ready: " + ", ".join(f"{k}={v}" for k, v in counts.items()))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the invoice review application.")
    parser.add_argument("--dataset", type=Path, default=Path("data/synthetic/seed-2026"))
    parser.add_argument("--db", type=Path, default=Path("data/review.db"))
    parser.add_argument("--model", type=Path, default=Path("models/pipeline.joblib"))
    parser.add_argument("--businesses", type=int, default=3,
                        help="how many businesses to load (0 for all)")
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--bootstrap", action="store_true",
                        help="generate data, render, train and seed before serving")
    parser.add_argument("--seed", action="store_true", help="re-seed the database only")
    parser.add_argument("--host", default=os.environ.get("HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", 8000)))
    parser.add_argument("--no-serve", action="store_true", help="set up only, do not serve")
    args = parser.parse_args(argv)

    os.environ["GSTPIPE_DATASET"] = str(args.dataset)
    os.environ["GSTPIPE_DB"] = str(args.db)
    os.environ["GSTPIPE_MODEL"] = str(args.model)

    if args.bootstrap:
        bootstrap(args.dataset, args.db, args.model, args.businesses, args.dpi)
    elif args.seed:
        from gstpipe.webapp.seed import seed_database
        counts = seed_database(args.dataset, args.db, args.businesses or None)
        print("seeded " + ", ".join(f"{k}={v}" for k, v in counts.items()))

    if args.no_serve:
        return 0

    if not args.db.exists():
        print(f"no database at {args.db}. Run with --bootstrap first.", file=sys.stderr)
        return 1

    import uvicorn
    print(f"serving on http://{args.host}:{args.port}")
    uvicorn.run("gstpipe.webapp.app:app", host=args.host, port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
