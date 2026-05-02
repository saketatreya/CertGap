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

import matplotlib.patches as patches

def draw_mechanism(ax):
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")
    ax.set_title("The Distribution Mismatch", fontsize=11, pad=10, fontweight="bold")
    
    # Draw states
    ax.add_patch(patches.Circle((2, 5), 0.8, facecolor="#4c78a8", edgecolor="black", linewidth=1.5, zorder=3))
    ax.add_patch(patches.Circle((5, 5), 0.8, facecolor="#e0e0e0", edgecolor="gray", linewidth=1.5, zorder=3))
    ax.add_patch(patches.Circle((8, 5), 0.8, facecolor="#e0e0e0", edgecolor="gray", linewidth=1.5, zorder=3))
    
    ax.text(2, 5, "$s_0$", ha="center", va="center", fontsize=12, color="white", fontweight="bold", zorder=4)
    ax.text(5, 5, "$s_1$", ha="center", va="center", fontsize=12, color="black", zorder=4)
    ax.text(8, 5, "$s_2$", ha="center", va="center", fontsize=12, color="black", zorder=4)
    
    # Draw transitions
    ax.annotate("", xy=(4.2, 5), xytext=(2.8, 5), arrowprops=dict(arrowstyle="->", lw=2, color="black"))
    ax.annotate("", xy=(7.2, 5), xytext=(5.8, 5), arrowprops=dict(arrowstyle="->", lw=2, color="black"))
    ax.annotate("...", xy=(9.5, 5), xytext=(8.8, 5), arrowprops=dict(arrowstyle="->", lw=2, color="black", ls="--"))
    
    # Label Start State
    ax.annotate("Improvement Target ($\\nu$)", xy=(2, 6.0), xytext=(2, 7.5),
                arrowprops=dict(facecolor='#4c78a8', shrink=0.05, width=1, headwidth=6),
                fontsize=10, color="black", ha="center", va="bottom")
                
    # Bracket for Rollout
    ax.plot([1.5, 1.5, 8.5, 8.5], [4.1, 3.8, 3.8, 4.1], color="gray", lw=1.5)
    ax.text(5, 3.4, "Rollout Distribution ($d^\\pi$)\n$\\epsilon_u$ minimized over all states",
            fontsize=10, color="gray", ha="center", va="top")

def plot_scatters(ax_res, ax_bias):
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
    ax_res.set_title("Absolute Residuals: Blind", fontsize=11, fontweight="bold")
    
    ax_bias.scatter(-dh, dj, alpha=0.3, s=10, color="#4c78a8")
    ax_bias.set_xlabel(r"Start-State Bias $-\hat\Delta_k$")
    ax_bias.set_title("Start-State Bias: Operative", fontsize=11, fontweight="bold")
    
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
    ax.set_ylabel("Median AUROC")
    ax.set_title("Audit Consensus", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.set_ylim(0.35, 0.8)

def main():
    fig = plt.figure(figsize=(15, 3.5))
    gs = fig.add_gridspec(1, 4)
    
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2], sharey=ax2)
    ax4 = fig.add_subplot(gs[0, 3])
    
    draw_mechanism(ax1)
    plot_scatters(ax2, ax3)
    plot_auroc_bars(ax4)
    
    plt.tight_layout()
    target = out_path("fig1_powerhouse_audit.pdf")
    plt.savefig(target)
    print(f"wrote {target}")

if __name__ == "__main__":
    main()
