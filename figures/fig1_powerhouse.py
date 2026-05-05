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
from certgap.analysis.harm_prediction import auroc
def out_path(name: str) -> Path:
    p = Path("figures/out") / name
    p.parent.mkdir(parents=True, exist_ok=True)
    return p

import matplotlib.patches as patches


# Per-seed AUROCs: same logic as scripts/build_table1.py, inlined so the
# figure script does not depend on the table-build script.
_ALGO_DIRS = [
    ("PPO",     "main", ["LunarLander-v3", "CartPole-v1", "Acrobot-v1",
                          "Hopper-v5", "HalfCheetah-v5", "Walker2d-v5",
                          "Ant-v5", "Humanoid-v5"]),
    ("PPO-MSE", "ppo_mse", ["LunarLander-v3", "Hopper-v5", "HalfCheetah-v5",
                             "Walker2d-v5", "Ant-v5", "Humanoid-v5"]),
    ("SAC",     "sac",     ["Hopper-v5", "HalfCheetah-v5", "Walker2d-v5",
                             "Ant-v5", "Humanoid-v5"]),
    ("TRPO",    "trpo",    ["LunarLander-v3", "Hopper-v5"]),
]


def _per_seed_auroc(algo, root, envs):
    eps_vals, dh_vals = [], []
    for env in envs:
        for p in sorted(Path(f"results/{root}/{env}").glob("seed_*.pkl")):
            with open(p, "rb") as f:
                d = pickle.load(f)
            if d["metadata"]["n_updates"] < 10:
                continue
            log = d["log"]
            h = np.asarray(log["harmful_k"], dtype=float)
            eps_vals.append(auroc(-np.asarray(log["eps_u"], dtype=float), h))
            dh_vals.append(auroc(-np.asarray(log["delta_hat_k"], dtype=float), h))
    return np.asarray(eps_vals, dtype=float), np.asarray(dh_vals, dtype=float)

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
    ax.annotate("True Objective ($\\nu$)\nEvaluated ONLY here", xy=(2, 6.0), xytext=(2, 7.5),
                arrowprops=dict(facecolor='#4c78a8', shrink=0.05, width=1, headwidth=6),
                fontsize=10, color="black", fontweight="bold", ha="center", va="bottom")
                
    # Bracket for Rollout
    ax.plot([1.5, 1.5, 8.5, 8.5], [4.1, 3.8, 3.8, 4.1], color="#e45756", lw=2)
    ax.text(5, 3.4, "Optimization Pressure ($d^\\pi$)\n$\\epsilon_u$ minimized over all states",
            fontsize=10, color="#e45756", fontweight="bold", ha="center", va="top")

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
    ax_res.set_title("Residual is Uninformative", fontsize=10, fontweight="bold")
    
    ax_bias.scatter(-dh, dj, alpha=0.3, s=10, color="#4c78a8")
    ax_bias.set_xlabel(r"Start-State Bias $-\hat\Delta_k$")
    ax_bias.set_title("Improvement controlled by Bias", fontsize=10, fontweight="bold")
    
    for ax, x, y in [(ax_res, res, dj), (ax_bias, -dh, dj)]:
        m, b = np.polyfit(x, y, 1)
        ax.plot(x, m*x + b, color="black", linestyle="--", linewidth=1)

def plot_auroc_bars(ax):
    algos = [a for a, _, _ in _ALGO_DIRS]
    colors = {"PPO": "#4c78a8", "PPO-MSE": "#54a24b", "SAC": "#f58518", "TRPO": "#e45756"}
    # paired-Wilcoxon dominance counts (from table1.csv)
    wins = {"PPO": "158/158", "PPO-MSE": "30/30", "SAC": "40/45", "TRPO": "19/19"}

    width = 0.36
    x = np.arange(len(algos))

    eps_vals_med, dh_vals_med = [], []
    eps_vals_all, dh_vals_all = [], []

    for algo, root, envs in _ALGO_DIRS:
        eps_seeds, dh_seeds = _per_seed_auroc(algo, root, envs)
        eps_vals_med.append(np.nanmedian(eps_seeds))
        dh_vals_med.append(np.nanmedian(dh_seeds))
        eps_vals_all.append(eps_seeds)
        dh_vals_all.append(dh_seeds)

    ax.bar(x - width/2, eps_vals_med, width, label=r"$-\epsilon_u$",
           color="lightgray", edgecolor="gray", linewidth=1.0, zorder=2)
    ax.bar(x + width/2, dh_vals_med, width, label=r"$-\hat\Delta_k$",
           color=[colors[a] for a in algos], edgecolor="black", linewidth=1.0, zorder=2)

    rng = np.random.default_rng(0)
    for i, algo in enumerate(algos):
        n = len(eps_vals_all[i])
        jitter_eps = rng.uniform(-width*0.35, width*0.35, size=n)
        jitter_dh = rng.uniform(-width*0.35, width*0.35, size=n)
        ax.scatter(np.full(n, x[i] - width/2) + jitter_eps, eps_vals_all[i],
                   s=12, color="black", alpha=0.35, zorder=3)
        ax.scatter(np.full(n, x[i] + width/2) + jitter_dh, dh_vals_all[i],
                   s=12, color="black", alpha=0.35, zorder=3)

    ax.axhline(0.5, color="black", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.text(len(algos) - 0.55, 0.505, "chance", fontsize=8, color="black", alpha=0.7,
            ha="right", va="bottom")

    for i, algo in enumerate(algos):
        ax.text(x[i], 0.235, f"$\\hat\\Delta$ wins\n{wins[algo]}",
                ha="center", va="bottom", fontsize=8, color="black")

    ax.set_xlim(-0.6, len(algos) - 0.4)
    ax.set_xticks(x)
    ax.set_xticklabels(algos, fontsize=10)
    ax.set_ylabel("Per-seed AUROC", fontsize=11)
    ax.set_title("Audit Consensus: $-\\hat\\Delta_k$ dominates $-\\epsilon_u$ on 262 of 267 paired seeds",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=10, loc="upper right", ncol=2, frameon=True)
    ax.set_ylim(0.20, 0.95)


def main():
    fig = plt.figure(figsize=(12, 6.5))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.15], hspace=0.45, wspace=0.30)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2], sharey=ax2)
    ax4 = fig.add_subplot(gs[1, :])  # wider bottom row for the headline panel

    draw_mechanism(ax1)
    plot_scatters(ax2, ax3)
    plot_auroc_bars(ax4)

    plt.tight_layout()
    target = out_path("fig1_powerhouse_audit.pdf")
    plt.savefig(target)
    print(f"wrote {target}")

if __name__ == "__main__":
    main()
