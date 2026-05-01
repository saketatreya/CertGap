"""Action B: Hyperparameter Selection Proof.

Plots mean final return vs mean concurrent delta_hat across the LunarLander 
factorial cells to prove the diagnostic is useful for model selection.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr

from figures._common import load_pickle, out_path, setup_matplotlib

def main() -> None:
    setup_matplotlib()
    
    base = Path("results/factorial_lunarlander")
    cells = [d for d in base.iterdir() if d.is_dir()]
    
    data = []
    for cell_dir in cells:
        seeds = list(cell_dir.glob("seed_*.pkl"))
        if not seeds: continue
        
        returns = []
        delta_hats = []
        for s in seeds:
            log = load_pickle(s)["log"]
            returns.append(np.mean(log["J_k"][-5:])) # Final performance
            delta_hats.append(np.nanmean(-log["delta_hat_k"])) # Diagnostic health
            
        data.append({
            "cell": cell_dir.name,
            "mean_return": np.mean(returns),
            "mean_delta_hat": np.mean(delta_hats)
        })
        
    df = pd.DataFrame(data)
    if df.empty: return
    
    rho, p = spearmanr(df["mean_delta_hat"], df["mean_return"])
    
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.scatter(df["mean_delta_hat"], df["mean_return"], color="#1f77b4", alpha=0.8, edgecolor="white", s=60)
    
    # Regression line
    m, b = np.polyfit(df["mean_delta_hat"], df["mean_return"], 1)
    x_range = np.array([df["mean_delta_hat"].min(), df["mean_delta_hat"].max()])
    ax.plot(x_range, m*x_range + b, color="black", linestyle="--", linewidth=1, alpha=0.5)
    
    ax.set_xlabel("Mean Concurrent Diagnostic $(-\hat\Delta_k)$", fontweight="bold")
    ax.set_ylabel("Final Mean Return", fontweight="bold")
    ax.set_title(f"Hyperparameter Selection (LunarLander)\nSpearman $\\rho = {rho:.2f}$", fontsize=10)
    
    plt.tight_layout()
    plt.savefig(out_path("fig_hyperparam_ranking.pdf"))
    print(f"wrote {out_path('fig_hyperparam_ranking.pdf')} (Spearman rho={rho:.2f})")

if __name__ == "__main__":
    main()
