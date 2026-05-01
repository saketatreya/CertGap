"""Residual-vs-improvement audit across algorithms."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from certgap.analysis.audit import collect_audit_rows
from figures._common import out_path, setup_matplotlib


def main() -> None:
    setup_matplotlib()
    rows = collect_audit_rows()
    rows = [row for row in rows if np.isfinite(row["auroc_neg_eps_u"])]
    if not rows:
        print("No residual audit data found.")
        return

    labels = [f"{row['algorithm']}\n{row['env'].split('-')[0]}" for row in rows]
    values = [row["auroc_neg_eps_u"] for row in rows]
    colors = ["#4c78a8" if row["algorithm"] == "PPO" else "#f58518" for row in rows]

    fig, ax = plt.subplots(figsize=(max(7, 0.35 * len(rows)), 3.5))
    x = np.arange(len(rows))
    ax.bar(x, values, color=colors, alpha=0.9, edgecolor="white", linewidth=0.6)
    ax.axhline(0.5, color="black", linestyle="--", linewidth=1.0, alpha=0.7)
    ax.set_ylabel("Median AUROC")
    ax.set_ylim(0.2, 0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_title("Bellman Residual vs. Policy Improvement")
    ax.grid(axis="y", linestyle=":", linewidth=0.8, alpha=0.4)
    plt.tight_layout()
    target = out_path("fig_residual_audit.pdf")
    plt.savefig(target)
    plt.close(fig)
    print(f"wrote {target}")


if __name__ == "__main__":
    main()
