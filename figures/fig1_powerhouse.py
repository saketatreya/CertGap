"""Figure 1: The Audit Powerhouse.
Contrast absolute residuals with start-state bias across algorithms.
"""

import pickle
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from certgap.analysis.audit import collect_audit_rows
def out_path(name: str) -> Path:
    p = Path("figures/out") / name
    p.parent.mkdir(parents=True, exist_ok=True)
    return p

def plot_scatters(ax_res, ax_bias):
    # Use Humanoid-v5 seed 0 as a representative case
    with open("results/main/Humanoid-v5/seed_0.pkl", "rb") as f:
        log = pickle.load(f)["log"]
        res = log["eps_u"]
        dh = log["delta_hat_k"]
        dj = log["delta_J_k"]
        mask = np.isfinite(res) & np.isfinite(dh) & np.isfinite(dj)
        res, dh, dj = res[mask], dh[mask], dj[mask]
        
    ax_res.scatter(res, dj, alpha=0.3, s=10, color="gray")
    ax_res.set_xlabel(r"Bellman Residual $|\epsilon_u|$")
    ax_res.set_ylabel(r"Actual Improvement $\Delta J$")
    ax_res.set_title("Absolute Residuals: Blind")
    
    ax_bias.scatter(-dh, dj, alpha=0.3, s=10, color="#4c78a8")
    ax_bias.set_xlabel(r"Start-State Bias $-\hat\Delta_k$")
    ax_bias.set_title("Start-State Bias: Informed")
    
    # Add regression lines
    for ax, x, y in [(ax_res, res, dj), (ax_bias, -dh, dj)]:
        m, b = np.polyfit(x, y, 1)
        ax.plot(x, m*x + b, color="black", linestyle="--", linewidth=1)

def plot_auroc_bars(ax):
    rows = collect_audit_rows()
    algos = ["PPO", "SAC", "TRPO"]
    colors = {"PPO": "#4c78a8", "SAC": "#f58518", "TRPO": "#e45756"}
    
    width = 0.25
    x = np.arange(len(algos))
    
    eps_vals = []
    dh_vals = []
    
    for algo in algos:
        a_rows = [r for r in rows if r["algorithm"] == algo]
        eps_vals.append(np.median([r["auroc_neg_eps_u"] for r in a_rows]))
        dh_vals.append(np.median([r["auroc_neg_delta_hat"] for r in a_rows]))
        
    ax.bar(x - width/2, eps_vals, width, label=r"$-\epsilon_u$", color="gray", alpha=0.4)
    ax.bar(x + width/2, dh_vals, width, label=r"$-\hat\Delta$", color=[colors[a] for a in algos])
    
    ax.axhline(0.5, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(algos)
    ax.set_ylabel("Median AUROC (Harm Prediction)")
    ax.set_title("Audit Consensus")
    ax.legend()
    ax.set_ylim(0.35, 0.8)

def main():
    fig = plt.figure(figsize=(12, 4))
    gs = fig.add_gridspec(1, 3)
    
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1], sharey=ax1)
    ax3 = fig.add_subplot(gs[0, 2])
    
    plot_scatters(ax1, ax2)
    plot_auroc_bars(ax3)
    
    plt.tight_layout()
    target = out_path("fig1_powerhouse_audit.pdf")
    plt.savefig(target)
    print(f"wrote {target}")

if __name__ == "__main__":
    main()
