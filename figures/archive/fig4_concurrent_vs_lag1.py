"""Figure 4: concurrent vs lag-1 AUROC. Diagnostic-controller boundary.

Tufte-style slopegraph showing the collapse of predictive power from concurrent to lag-1.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from certgap.analysis.harm_prediction import auroc
from figures._common import (
    ENV_ORDER,
    env_seed_paths,
    load_pickle,
    out_path,
    setup_matplotlib,
)


def per_env_pair(env: str) -> tuple[float, float]:
    paths = env_seed_paths(env)
    if not paths:
        return float("nan"), float("nan")
    cg_chunks: list[np.ndarray] = []
    harm_chunks: list[np.ndarray] = []
    for path in paths:
        log = load_pickle(path)["log"]
        cg_chunks.append(np.asarray(log["cert_gap_k"], dtype=float))
        harm_chunks.append(np.asarray(log["harmful_k"], dtype=float))
    cg = np.concatenate(cg_chunks)
    harm = np.concatenate(harm_chunks)
    concurrent = auroc(cg, harm)

    cg_lag: list[np.ndarray] = []
    h_lag: list[np.ndarray] = []
    for c, h in zip(cg_chunks, harm_chunks):
        if c.size < 2:
            continue
        cg_lag.append(c[:-1])
        h_lag.append(h[1:])
    if cg_lag:
        lag1 = auroc(np.concatenate(cg_lag), np.concatenate(h_lag))
    else:
        lag1 = float("nan")
    return concurrent, lag1


def main() -> None:
    setup_matplotlib()
    fig, ax = plt.subplots(figsize=(6, 5.5))
    cmap = plt.get_cmap("tab10")
    
    # Remove all spines for a clean Tufte look
    for spine in ax.spines.values():
        spine.set_visible(False)
        
    ax.axhline(0.5, color="gray", linewidth=1.0, linestyle="--", alpha=0.5)
    ax.text(0.5, 0.505, "Random Chance (0.5)", color="gray", ha="center", fontsize=9, fontweight="bold")
    
    for i, env in enumerate(ENV_ORDER):
        concurrent, lag1 = per_env_pair(env)
        if np.isnan(concurrent):
            continue
        env_name = env.split("-")[0]
        color = cmap(i % 10)
        
        # Plot the line
        ax.plot([0, 1], [concurrent, lag1], marker="o", color=color, linewidth=2.0, markersize=6)
        
        # Label left (Concurrent)
        ax.text(-0.03, concurrent, f"{env_name} ({concurrent:.2f})", color=color, ha="right", va="center", fontsize=9, fontweight="bold")
        # Label right (Lag-1)
        ax.text(1.03, lag1, f"({lag1:.2f})", color=color, ha="left", va="center", fontsize=9, fontweight="bold")

    ax.set_xticks([0, 1])
    ax.set_xticklabels([
        "Concurrent Prediction\n(Metric at step $k$ predicts harm at step $k$)",
        "Lag-1 Prediction\n(Metric at step $k$ predicts harm at step $k+1$)"
    ], fontweight="bold")
    
    # Hide the y-axis ticks and labels since we label points directly
    ax.set_yticks([])
    ax.set_ylabel("")
    
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(0.2, 1.05)
    
    plt.tight_layout()
    plt.savefig(out_path("fig4_concurrent_vs_lag1.pdf"))
    plt.close(fig)
    print(f"wrote {out_path('fig4_concurrent_vs_lag1.pdf')}")


if __name__ == "__main__":
    main()
