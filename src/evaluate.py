import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def normalize_output(val) -> str:
    """Normalizes numeric strings (e.g., '1539.50' vs '1539.5') so determinism checks measure semantic drift, not trailing zeros."""
    if pd.isna(val) or str(val).strip() == "":
        return "INVALID"
    s = str(val).strip()
    try:
        return f"{float(s):.4f}".rstrip("0").rstrip(".")
    except ValueError:
        return s


def compute_determinism(df: pd.DataFrame) -> pd.DataFrame:
    """Computes strict multi-run agreement per (model, tier, task_id)."""
    df = df.copy()
    df["norm_answer"] = df["extracted_answer"].apply(normalize_output)

    grouped = (
        df.groupby(["model", "tier", "domain", "task_id"])
        .agg(
            unique_answers=("norm_answer", "nunique"),
            any_invalid=("norm_answer", lambda x: "INVALID" in set(x)),
            mean_accuracy=("is_correct", "mean"),
        )
        .reset_index()
    )
    # A task is strictly deterministic if all N runs produced the exact same valid answer
    grouped["is_deterministic"] = (grouped["unique_answers"] == 1) & (~grouped["any_invalid"])
    return grouped


def generate_plots(summary_df: pd.DataFrame, domain_df: pd.DataFrame, out_dir: Path):
    """Generates 300-DPI publication figures for Buletinul Stiintific UPB."""
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10})

    tiers = summary_df["tier"].tolist()
    x = np.arange(len(tiers))
    width = 0.26

    # Figure 1: Grouped Bar Chart (Schema Compliance, Pass@1 Accuracy, Multi-Run Determinism)
    fig, ax = plt.subplots(figsize=(8.5, 4.5), dpi=300)
    b1 = ax.bar(x - width, summary_df["json_compliance_pct"], width, label="JSON Schema Compliance (%)", color="#2b5c8f")
    b2 = ax.bar(x, summary_df["accuracy_pct"], width, label="Task Accuracy (%)", color="#388e3c")
    b3 = ax.bar(x + width, summary_df["determinism_pct"], width, label="Multi-Run Determinism (%)", color="#d97706")

    ax.set_ylabel("Percentage (%)")
    ax.set_title("Impact of Prompt Architecture on Reliability, Accuracy, and Determinism")
    ax.set_xticks(x)
    ax.set_xticklabels(tiers)
    ax.set_ylim(0, 115)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend(frameon=True, loc="upper left")

    for bars in (b1, b2, b3):
        for bar in bars:
            h = bar.get_height()
            ax.annotate(
                f"{h:.1f}%",
                xy=(bar.get_x() + bar.get_width() / 2, h),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    fig.tight_layout()
    fig.savefig(out_dir / "fig1_prompt_tier_comparison.png", dpi=300)
    plt.close(fig)

    # Figure 2: Domain-by-Tier Accuracy Breakdown
    pivot = domain_df.pivot(index="tier", columns="domain", values="accuracy_pct").reindex(tiers)
    fig, ax = plt.subplots(figsize=(8.0, 4.2), dpi=300)
    pivot.plot(kind="bar", ax=ax, width=0.72, colormap="viridis", edgecolor="black", linewidth=0.5)
    ax.set_ylabel("Task Accuracy (%)")
    ax.set_xlabel("Prompt Architecture Tier")
    ax.set_title("Task Accuracy Across Engineering Domains by Prompt Tier")
    ax.set_ylim(0, 115)
    ax.tick_params(axis="x", rotation=0)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend(title="Engineering Domain", frameon=True)

    fig.tight_layout()
    fig.savefig(out_dir / "fig2_domain_accuracy_breakdown.png", dpi=300)
    plt.close(fig)


def evaluate_results(csv_paths: list[str], figures_dir: str):
    frames = [pd.read_csv(p) for p in csv_paths]
    df = pd.concat(frames, ignore_index=True)

    det_df = compute_determinism(df)

    # Overall summary per (model, tier)
    run_metrics = (
        df.groupby(["model", "tier"])
        .agg(
            total_runs=("is_correct", "count"),
            json_compliance_pct=("valid_json", lambda x: round(x.mean() * 100, 1)),
            accuracy_pct=("is_correct", lambda x: round(x.mean() * 100, 1)),
            avg_latency_ms=("latency_ms", lambda x: round(x.mean(), 1)),
            avg_completion_tokens=("completion_tokens", lambda x: round(x.mean(), 1)),
        )
        .reset_index()
    )

    det_metrics = (
        det_df.groupby(["model", "tier"])
        .agg(determinism_pct=("is_deterministic", lambda x: round(x.mean() * 100, 1)))
        .reset_index()
    )

    summary = pd.merge(run_metrics, det_metrics, on=["model", "tier"])
    summary.to_csv("data/results/summary_metrics.csv", index=False)

    # Domain breakdown
    domain_summary = (
        df.groupby(["model", "tier", "domain"])
        .agg(
            accuracy_pct=("is_correct", lambda x: round(x.mean() * 100, 1)),
            json_compliance_pct=("valid_json", lambda x: round(x.mean() * 100, 1)),
        )
        .reset_index()
    )
    domain_summary.to_csv("data/results/domain_metrics.csv", index=False)

    print("\n=== OVERALL BENCHMARK SUMMARY ===")
    print(summary.to_markdown(index=False))

    print("\n=== DOMAIN BREAKDOWN ===")
    print(domain_summary.to_markdown(index=False))

    # Generate plots for the primary model in the dataset
    primary_model = summary["model"].iloc[0]
    generate_plots(
        summary[summary["model"] == primary_model],
        domain_summary[domain_summary["model"] == primary_model],
        Path(figures_dir),
    )
    print(f"\nExported 300-DPI figures to {figures_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate LLM benchmark CSV logs and generate UPB paper figures")
    parser.add_argument(
        "--inputs",
        nargs="+",
        default=["data/results/openai_gpt-oss-20b_pilot.csv"],
        help="One or more result CSV files to analyze",
    )
    parser.add_argument("--figures-dir", default="paper/figures", help="Output directory for high-DPI plots")
    args = parser.parse_args()

    evaluate_results(args.inputs, args.figures_dir)