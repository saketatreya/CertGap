"""Figure 3: the right axis is Δ̂_k, not ε_u.

Two panels.

Left:  (A_k, ε_u) plane, colored by sign of ΔJ_k.
       The standard improvement bound predicts beneficial in the region
       A_k > C·ε_u. Updates do not separate cleanly by this axis.

Right: (A_k, Δ̂_k) plane, colored by sign of ΔJ_k.
       The exact identity (Prop 1) makes the boundary the diagonal
       A_k = Δ̂_k. Beneficial updates lie strictly below the diagonal,
       harmful strictly above. The boundary is sharp.

Pooled across the PPO main grid (8 envs × 20 seeds, ~24,500 updates),
within-env z-scored so every environment contributes comparably.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from figures._common import out_path, setup_matplotlib

ENV_ORDER = [
    "LunarLander-v3", "CartPole-v1", "Acrobot-v1",
    "Hopper-v5", "HalfCheetah-v5", "Walker2d-v5", "Ant-v5", "Humanoid-v5",
]


def collect():
    """Return per-update (A, eps, dhat, harm) z-scored within env."""
    a_chunks, eps_chunks, dh_chunks, h_chunks = [], [], [], []
    for env in ENV_ORDER:
        env_a, env_e, env_d, env_h = [], [], [], []
        for p in sorted(Path(f"results/main/{env}").glob("seed_*.pkl")):
            with open(p, "rb") as f:
                log = pickle.load(f)["log"]
            a = np.asarray(log["A_k"], dtype=float)
            e = np.asarray(log["eps_u"], dtype=float)
            d = np.asarray(log["delta_hat_k"], dtype=float)
            h = np.asarray(log["harmful_k"], dtype=float)
            mask = np.isfinite(a) & np.isfinite(e) & np.isfinite(d) & np.isfinite(h)
            env_a.append(a[mask]); env_e.append(e[mask]); env_d.append(d[mask]); env_h.append(h[mask])
        if not env_a:
            continue
        env_a = np.concatenate(env_a); env_e = np.concatenate(env_e)
        env_d = np.concatenate(env_d); env_h = np.concatenate(env_h)
        # within-env z-score for A, eps, dhat (so envs are comparable in plot scale)
        for arr in (env_a, env_e, env_d):
            mu = arr.mean(); sd = max(arr.std(), 1e-9)
            arr -= mu; arr /= sd
        a_chunks.append(env_a); eps_chunks.append(env_e)
        dh_chunks.append(env_d); h_chunks.append(env_h)
    return (
        np.concatenate(a_chunks),
        np.concatenate(eps_chunks),
        np.concatenate(dh_chunks),
        np.concatenate(h_chunks),
    )


def panel(ax, x, y, harm, *, xlabel, ylabel, title, draw_diagonal: bool):
    beneficial = harm < 0.5
    harmful = ~beneficial
    # subsample for readability
    rng = np.random.default_rng(0)
    keep_b = rng.choice(np.arange(beneficial.sum()), size=min(beneficial.sum(), 5000), replace=False)
    keep_h = rng.choice(np.arange(harmful.sum()), size=min(harmful.sum(), 5000), replace=False)
    xb = x[beneficial][keep_b]; yb = y[beneficial][keep_b]
    xh = x[harmful][keep_h];   yh = y[harmful][keep_h]

    ax.scatter(xb, yb, s=4, alpha=0.20, color="#2ca02c", linewidths=0,
               label=r"beneficial ($\Delta J\geq 0$)")
    ax.scatter(xh, yh, s=4, alpha=0.30, color="#d62728", linewidths=0,
               label=r"harmful ($\Delta J<0$)")
    if draw_diagonal:
        lo, hi = -3.5, 3.5
        ax.plot([lo, hi], [lo, hi], color="black", linestyle="--", linewidth=1.2,
                label="identity boundary $A_k=\\hat\\Delta_k$", zorder=4)
    ax.axhline(0, color="0.7", linewidth=0.4)
    ax.axvline(0, color="0.7", linewidth=0.4)
    ax.set_xlim(-3.5, 3.5)
    ax.set_ylim(-3.5, 3.5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)


def main() -> None:
    setup_matplotlib()
    A, E, D, H = collect()
    n = A.size
    print(f"pooled updates: {n}")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))

    # Left: A vs eps_u — practitioner's view, no clean boundary
    panel(
        axes[0], A, E, H,
        xlabel=r"$A_k$ (z-scored within env)",
        ylabel=r"$\varepsilon_u$ (z-scored within env)",
        title="Standard view: $\\varepsilon_u$ is the wrong axis",
        draw_diagonal=False,
    )
    axes[0].legend(loc="upper left", frameon=False, fontsize=8, markerscale=2)

    # Right: A vs dhat — identity makes diagonal the exact boundary
    panel(
        axes[1], A, D, H,
        xlabel=r"$A_k$ (z-scored within env)",
        ylabel=r"$\hat\Delta_k$ (z-scored within env)",
        title="Identity view: $A_k = \\hat\\Delta_k$ is the exact boundary",
        draw_diagonal=True,
    )
    axes[1].legend(loc="upper left", frameon=False, fontsize=8, markerscale=2)

    plt.suptitle(
        f"Pooled PPO updates across 8 environments × 20 seeds ($n\\approx{n//1000}$k)",
        fontsize=10, y=1.02,
    )
    plt.tight_layout()
    plt.savefig(out_path("fig3_decomp_planes.pdf"))
    plt.close(fig)
    print(f"wrote {out_path('fig3_decomp_planes.pdf')}")


if __name__ == "__main__":
    main()
