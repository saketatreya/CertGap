"""Figure 2: Mechanism and Intervention.
The Ak Paradox and the Early Stopping Pareto.
"""

import pickle
from pathlib import Path
import numpy as np
import matplotlib as mpl
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt
import pandas as pd
def out_path(name: str) -> Path:
    p = Path("figures/out") / name
    p.parent.mkdir(parents=True, exist_ok=True)
    return p

def collect_env_data(env):
    """Concatenate (A_k, delta_hat_k, delta_J_k) across all seeds of an env."""
    all_A, all_dh, all_dj = [], [], []
    root = Path(f"results/main/{env}")
    for p in root.glob("seed_*.pkl"):
        with p.open("rb") as f:
            log = pickle.load(f)["log"]
            mask = np.isfinite(log["A_k"]) & np.isfinite(log["delta_hat_k"]) & np.isfinite(log["delta_J_k"])
            all_A.extend(log["A_k"][mask])
            all_dh.extend(log["delta_hat_k"][mask])
            all_dj.extend(log["delta_J_k"][mask])
    return np.array(all_A), np.array(all_dh), np.array(all_dj)


def collect_humanoid_data():
    return collect_env_data("Humanoid-v5")


def _pareto_xy(dh, dj, n_points=50):
    thresholds = np.linspace(np.percentile(dh, 5), np.percentile(dh, 95), n_points)
    harmful_total = (dj < 0).sum()
    imp_total = dj[dj > 0].sum()
    hp = np.array([((dj < 0) & (dh > t)).sum() / harmful_total for t in thresholds])
    ik = np.array([1.0 - (dj[(dj > 0) & (dh > t)].sum() / imp_total) for t in thresholds])
    return hp, ik


def plot_ak_paradox(ax1, ax2):
    A, dh, dj = collect_humanoid_data()

    ax1.hexbin(A, dj, gridsize=25, cmap='Reds', mincnt=1, alpha=0.8)
    ax1.set_xlabel(r"Surrogate Advantage $A_k$")
    ax1.set_ylabel(r"Actual Improvement $\Delta J$")
    ax1.set_title("The $A_k$ Paradox")

    ax2.hexbin(A, dh, gridsize=25, cmap='Blues', mincnt=1, alpha=0.8)
    ax2.set_xlabel(r"Surrogate Advantage $A_k$")
    ax2.set_ylabel(r"Start-State Bias $\hat\Delta_k$")
    ax2.set_title("The Overfitting Coupling")


def plot_pareto(ax):
    """Pareto frontier of post-hoc threshold sweep on hat-Delta_k.

    Overlays the canonical high-variance env (Humanoid-v5) and a second env
    (Hopper-v5) to show the gate generalises beyond a single environment.
    """
    envs = [
        ("Humanoid-v5", "#2ca02c", "o", "Humanoid"),
        ("Hopper-v5",   "#d62728", "s", "Hopper"),
    ]
    for env, color, marker, label in envs:
        A, dh, dj = collect_env_data(env)
        if len(dh) == 0:
            continue
        hp, ik = _pareto_xy(dh, dj)
        ax.plot(hp, ik, linestyle="-", marker=marker, markersize=3.5,
                color=color, label=label, alpha=0.85)

    ax.set_xlabel("Harmful Updates Prevented")
    ax.set_ylabel("Improvement Retained")
    ax.set_title("Intervention Pareto (post-hoc gate)")
    ax.grid(True, alpha=0.2)
    ax.legend(loc="lower left", fontsize=9, frameon=True)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)

def main():
    fig = plt.figure(figsize=(12, 4))
    gs = fig.add_gridspec(1, 3)
    
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    
    plot_ak_paradox(ax1, ax2)
    plot_pareto(ax3)
    
    plt.tight_layout()
    target = out_path("fig2_mechanism_intervention.pdf")
    plt.savefig(target)
    print(f"wrote {target}")

if __name__ == "__main__":
    main()
