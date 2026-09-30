"""Figures for the propagation study."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

HEADLINE = {"1_extraction": "field_accuracy", "2_categorisation": "accuracy",
            "3_reconciliation": "f1", "4_anomaly": "f1"}
STAGE_LABEL = {"1_extraction": "1. Extraction (input quality)", "2_categorisation": "2. Categorisation",
               "3_reconciliation": "3. Reconciliation", "4_anomaly": "4. Anomaly detection"}
BEST = {"1_extraction": "injected", "2_categorisation": "hybrid_lr",
        "3_reconciliation": "m3_forest", "4_anomaly": "a3_isolation_forest"}
COLOURS = {"1_extraction": "#5b6472", "2_categorisation": "#2e7d4f",
           "3_reconciliation": "#2f6fed", "4_anomaly": "#d94848"}


def _headline(summary: pd.DataFrame) -> pd.DataFrame:
    mask = summary.apply(lambda r: HEADLINE.get(r.stage) == r.metric, axis=1)
    return summary[mask]


def degradation_curves(summary: pd.DataFrame, path: Path) -> Path:
    data = _headline(summary)
    fig, ax = plt.subplots(figsize=(8, 5), dpi=160)
    for stage, label in STAGE_LABEL.items():
        rows = data[(data.stage == stage) & (data.method == BEST[stage])].sort_values("rate")
        if rows.empty:
            continue
        ax.errorbar(rows.rate * 100, rows["mean"], yerr=rows["std"].fillna(0), marker="o",
                    capsize=3, linewidth=2, color=COLOURS[stage], label=label)
    ax.set_xlabel("Injected extraction error rate (% of fields)")
    ax.set_ylabel("Headline metric (accuracy or F1)")
    ax.set_title("How far an extraction error travels")
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def amplification_chart(summary: pd.DataFrame, path: Path) -> Path:
    data = _headline(summary)
    data = data[data.rate > 0]
    fig, ax = plt.subplots(figsize=(8, 5), dpi=160)
    for stage, label in STAGE_LABEL.items():
        if stage == "1_extraction":
            continue
        rows = data[(data.stage == stage) & (data.method == BEST[stage])].sort_values("rate")
        if rows.empty:
            continue
        ax.plot(rows.rate * 100, rows.amplification, marker="o", linewidth=2,
                color=COLOURS[stage], label=label)
    ax.axhline(1.0, color="#333333", linestyle="--", linewidth=1)
    ax.text(30, 1.06, "above this line a stage amplifies upstream error", fontsize=8,
            ha="right", color="#333333")
    ax.set_xlabel("Injected extraction error rate (% of fields)")
    ax.set_ylabel("Amplification factor  A(s, e)")
    ax.set_title("Which stages absorb error, and which amplify it")
    ax.set_yscale("log")
    ax.grid(alpha=0.25, which="both")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def anomaly_mechanism(summary: pd.DataFrame, path: Path) -> Path:
    rows = summary[(summary.stage == "4_anomaly") & (summary.method == BEST["4_anomaly"])]
    fig, ax = plt.subplots(figsize=(8, 5), dpi=160)
    styles = {"precision": ("#d94848", "-"), "recall": ("#2f6fed", "-"),
              "flagged_rate": ("#5b6472", "--")}
    for metric, (colour, style) in styles.items():
        series = rows[rows.metric == metric].sort_values("rate")
        if series.empty:
            continue
        ax.plot(series.rate * 100, series["mean"], marker="o", linewidth=2, linestyle=style,
                color=colour, label=metric.replace("_", " "))
    ax.set_xlabel("Injected extraction error rate (% of fields)")
    ax.set_ylabel("Rate")
    ax.set_title("Why anomaly detection fails: noise looks like an anomaly")
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def write_all(summary: pd.DataFrame, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    return [degradation_curves(summary, out_dir / "degradation_curves.png"),
            amplification_chart(summary, out_dir / "amplification.png"),
            anomaly_mechanism(summary, out_dir / "anomaly_mechanism.png")]
