"""Figure 1: The Audit Powerhouse.
Contrast absolute residuals with start-state bias across algorithms.
"""

import pickle
from pathlib import Path
import numpy as np
import matplotlib as mpl
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt
from certgap.analysis.audit import collect_audit_rows
def out_path(name: str) -> Path:
    p = Path("figures/out") / name
    p.parent.mkdir(parents=True, exist_ok=True)
    return p

from matplotlib.patches import Ellipse
import matplotlib.patches as patches

def draw_mechanism(ax):
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")
    ax.set_title("The Distribution Mismatch", fontsize=12, pad=10, fontweight="bold")
    
    # Draw d^\pi (Rollout Distribution)
    d_pi = Ellipse((5, 5), width=8, height=6, angle=15, 
                   facecolor="#e0e0e0", edgecolor="gray", linewidth=2, alpha=0.5)
    ax.add_patch(d_pi)
    ax.text(8.5, 6.5, r"Rollout Dist. $d^\pi$", fontsize=11, color="gray", ha="center")
    
    # Draw \nu (Start-State Distribution)
    nu = Ellipse((3.5, 4.5), width=2.5, height=1.5, angle=0, 
                 facecolor="#4c78a8", edgecolor="black", linewidth=2, alpha=0.7)
    ax.add_patch(nu)
    ax.text(3.5, 4.3, r"Start States $\nu$", fontsize=11, color="white", ha="center", fontweight="bold")
    
    # Arrow for Bellman Residual
    ax.annotate(r"$\epsilon_u$ minimized here", xy=(6.5, 3), xytext=(8, 1.5),
                arrowprops=dict(facecolor='gray', shrink=0.05, width=1, headwidth=6),
                fontsize=10, color="gray", ha="center")
                
    # Arrow for Policy Improvement
    ax.annotate(r"Improvement relies on here", xy=(3.5, 5.5), xytext=(2, 8),
                arrowprops=dict(facecolor='#4c78a8', shrink=0.05, width=1, headwidth=6),
                fontsize=10, color="black", ha="center")

def plot_scatter_bias(ax_bias):
    # Use Humanoid-v5 seed 0 as a representative case
    with open("results/main/Humanoid-v5/seed_0.pkl", "rb") as f:
        log = pickle.load(f)["log"]
        dh = log["delta_hat_k"]
        dj = log["delta_J_k"]
        mask = np.isfinite(dh) & np.isfinite(dj)
        dh, dj = dh[mask], dj[mask]
        
    ax_bias.scatter(-dh, dj, alpha=0.3, s=10, color="#4c78a8")
    ax_bias.set_xlabel(r"Start-State Bias $-\hat\Delta_k$")
    ax_bias.set_ylabel(r"Actual Improvement $\Delta J$")
    ax_bias.set_title("Start-State Bias: Operative", fontweight="bold")
    
    # Add regression line
    m, b = np.polyfit(-dh, dj, 1)
    ax_bias.plot(-dh, m*(-dh) + b, color="black", linestyle="--", linewidth=1)

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
    ax.set_title("Audit Consensus", fontweight="bold")
    ax.legend()
    ax.set_ylim(0.35, 0.8)

def main():
    fig = plt.figure(figsize=(12, 4))
    gs = fig.add_gridspec(1, 3)
    
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    
    draw_mechanism(ax1)
    plot_scatter_bias(ax2)
    plot_auroc_bars(ax3)
    
    plt.tight_layout()
    target = out_path("fig1_powerhouse_audit.pdf")
    plt.savefig(target)
    print(f"wrote {target}")

if __name__ == "__main__":
    main()
