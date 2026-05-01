"""Figure 1: Bellman residual audit across actor--critic families.

Groups results by algorithm (PPO, SAC, TRPO) and compares -eps_u vs -delta_hat.
"""

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
    if not rows:
        print("No audit data found.")
        return

    # Group by algorithm
    algos = sorted(list(set(row["algorithm"] for row in rows)))
    # Ensure PPO is first if present
    if "PPO" in algos:
        algos.remove("PPO")
        algos = ["PPO"] + algos

    fig, axes = plt.subplots(1, len(algos), figsize=(3.5 * len(algos), 3.8), sharey=True)
    if len(algos) == 1:
        axes = [axes]

    colors = {"PPO": "#4c78a8", "PPO-MSE": "#72b7b2", "SAC": "#f58518", "TRPO": "#e45756"}

    for i, algo in enumerate(algos):
        ax = axes[i]
        algo_rows = [row for row in rows if row["algorithm"] == algo]
        
        # Sort envs by paper order
        from certgap.analysis.audit import ENV_ORDER
        algo_rows = sorted(algo_rows, key=lambda r: ENV_ORDER.index(r["env"]) if r["env"] in ENV_ORDER else 99)
        
        envs = [row["env"].split("-")[0] for row in algo_rows]
        eps_u = [row["auroc_neg_eps_u"] for row in algo_rows]
        delta_hat = [row["auroc_neg_delta_hat"] for row in algo_rows]
        
        x = np.arange(len(envs))
        width = 0.35
        
        ax.bar(x - width/2, eps_u, width, label=r"$-\epsilon_u$", color="gray", alpha=0.4, edgecolor="white", linewidth=0.5)
        ax.bar(x + width/2, delta_hat, width, label=r"$-\hat\Delta$", color=colors.get(algo, "#555555"), edgecolor="white", linewidth=0.5)
        
        ax.axhline(0.5, color="black", linestyle="--", linewidth=1.0, alpha=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels(envs, rotation=40, ha="right", fontsize=9)
        ax.set_title(r"\textbf{" + algo + r"}", fontsize=11)
        ax.set_ylim(0.25, 0.85)
        if i == 0:
            ax.set_ylabel("Median AUROC", fontweight="bold")
            ax.legend(loc="upper left", frameon=False, fontsize=9)
        ax.grid(axis="y", linestyle=":", alpha=0.4)

    plt.tight_layout()
    target = out_path("fig1_auroc_bars.pdf")
    plt.savefig(target)
    plt.close(fig)
    print(f"wrote {target}")

if __name__ == "__main__":
    main()
