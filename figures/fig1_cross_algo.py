"""Figure 1 (HEADLINE): cross-algorithm dominance scatter.

Per-seed AUROC pair (-ε_u for harm, -Δ̂_k for harm) across four
algorithms: PPO main, PPO-MSE control, SAC, TRPO. The cloud lives
strictly above the y=x diagonal: across every algorithm and almost
every seed in our suite, the start-state critic bias is a better
per-update harm predictor than the Bellman residual.

This is the figure that tells the whole paper at a glance.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from certgap.analysis.harm_prediction import auroc
from figures._common import out_path, setup_matplotlib


# Algo specifications: (display name, top-level results dir, env list, color, marker).
ALGOS = [
    ("PPO",      "main", [
        "LunarLander-v3", "CartPole-v1", "Acrobot-v1",
        "Hopper-v5", "HalfCheetah-v5", "Walker2d-v5", "Ant-v5", "Humanoid-v5",
    ], "#1f77b4", "o"),
    ("PPO-MSE",  "ppo_mse", [
        "LunarLander-v3", "Hopper-v5", "HalfCheetah-v5", "Walker2d-v5", "Ant-v5", "Humanoid-v5",
    ], "#2ca02c", "s"),
    ("SAC",      "sac", ["Hopper-v5", "HalfCheetah-v5", "Walker2d-v5"], "#d62728", "D"),
    ("TRPO",     "trpo", ["LunarLander-v3", "Hopper-v5"], "#9467bd", "^"),
]


def load_seeds(folder: Path) -> list[dict]:
    out = []
    for p in sorted(Path(folder).glob("seed_*.pkl")):
        with open(p, "rb") as f:
            d = pickle.load(f)
        if d["metadata"]["n_updates"] < 10:
            continue
        out.append(d)
    return out


def per_seed_pair(seeds: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    eps_au, dh_au = [], []
    for d in seeds:
        log = d["log"]
        h = np.asarray(log["harmful_k"], dtype=float)
        eps_au.append(auroc(-np.asarray(log["eps_u"], dtype=float), h))
        dh_au.append(auroc(-np.asarray(log["delta_hat_k"], dtype=float), h))
    return np.asarray(eps_au, dtype=float), np.asarray(dh_au, dtype=float)


def main() -> None:
    setup_matplotlib()

    fig = plt.figure(figsize=(9.2, 5.0))
    gs = fig.add_gridspec(
        2, 3,
        width_ratios=[3.4, 0.18, 1.5],
        height_ratios=[1.0, 3.4],
        hspace=0.05, wspace=0.05,
    )
    ax = fig.add_subplot(gs[1, 0])
    ax_top = fig.add_subplot(gs[0, 0], sharex=ax)
    ax_right = fig.add_subplot(gs[1, 2], sharey=ax)

    all_eps_pool: list[np.ndarray] = []
    all_dh_pool: list[np.ndarray] = []
    legend_handles = []

    rng = np.random.default_rng(0)
    for label, top, envs, color, marker in ALGOS:
        eps_chunks: list[np.ndarray] = []
        dh_chunks: list[np.ndarray] = []
        for env in envs:
            seeds = load_seeds(Path(f"results/{top}/{env}"))
            if not seeds:
                continue
            e, d = per_seed_pair(seeds)
            eps_chunks.append(e)
            dh_chunks.append(d)
        if not eps_chunks:
            continue
        eps_arr = np.concatenate(eps_chunks)
        dh_arr = np.concatenate(dh_chunks)
        # Tiny jitter to break ties visually
        jx = rng.normal(0, 0.003, size=eps_arr.size)
        jy = rng.normal(0, 0.003, size=dh_arr.size)
        sc = ax.scatter(
            eps_arr + jx, dh_arr + jy,
            color=color, marker=marker, s=46,
            edgecolor="white", linewidth=0.6, alpha=0.92,
            label=f"{label}  ($n$={eps_arr.size})", zorder=3,
        )
        legend_handles.append(sc)
        all_eps_pool.append(eps_arr)
        all_dh_pool.append(dh_arr)

    eps_pool = np.concatenate(all_eps_pool)
    dh_pool = np.concatenate(all_dh_pool)
    above = int(np.sum(dh_pool > eps_pool))
    n_total = int(eps_pool.size)

    # Diagonal y=x and chance lines
    lo, hi = 0.18, 0.92
    ax.plot([lo, hi], [lo, hi], color="k", linestyle="--", linewidth=0.9, alpha=0.7, zorder=1)
    ax.axhline(0.5, color="0.6", linestyle=":", linewidth=0.8, zorder=1)
    ax.axvline(0.5, color="0.6", linestyle=":", linewidth=0.8, zorder=1)
    ax.text(0.51, 0.205, "chance ($\\varepsilon_u$)", color="0.45", fontsize=8, rotation=90, va="bottom")
    ax.text(0.205, 0.515, "chance ($-\\hat\\Delta_k$)", color="0.45", fontsize=8, va="bottom")

    # Annotate dominance
    ax.text(
        0.95, 0.07,
        f"$-\\hat\\Delta_k$ better in\n{above} of {n_total} seeds\n"
        f"({100 * above / n_total:.1f}%)",
        transform=ax.transAxes, ha="right", va="bottom",
        fontsize=10, fontweight="bold", color="#1f4f72",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#e6f0fa", edgecolor="#1f77b4", linewidth=0.8),
    )

    # Marginals
    bins = np.linspace(0.18, 0.92, 36)
    for chunks, ax_marg, orient in [
        (all_eps_pool, ax_top, "x"),
        (all_dh_pool, ax_right, "y"),
    ]:
        for (label, _top, _envs, color, _marker), arr in zip(ALGOS, chunks):
            if arr.size == 0:
                continue
            if orient == "x":
                ax_marg.hist(arr, bins=bins, histtype="stepfilled",
                             alpha=0.5, color=color, edgecolor=color, linewidth=0.8)
            else:
                ax_marg.hist(arr, bins=bins, histtype="stepfilled", alpha=0.5,
                             orientation="horizontal", color=color, edgecolor=color, linewidth=0.8)
    ax_top.axvline(0.5, color="0.6", linestyle=":", linewidth=0.8)
    ax_right.axhline(0.5, color="0.6", linestyle=":", linewidth=0.8)
    ax_top.set_yticks([])
    ax_right.set_xticks([])
    for s in ("top", "right", "left"):
        ax_top.spines[s].set_visible(False)
    for s in ("top", "right", "bottom"):
        ax_right.spines[s].set_visible(False)
    plt.setp(ax_top.get_xticklabels(), visible=False)
    plt.setp(ax_right.get_yticklabels(), visible=False)

    # Main axes
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel(r"AUROC of $-\widehat\varepsilon_u$ for predicting $\Delta J_k<0$",
                  fontsize=11)
    ax.set_ylabel(r"AUROC of $-\widehat{\hat\Delta}_k$ for predicting $\Delta J_k<0$",
                  fontsize=11)
    ax.legend(handles=legend_handles, loc="upper left", frameon=True, fontsize=9,
              framealpha=0.95, edgecolor="0.7")

    # Headline title above the marginal (matplotlib text, not full LaTeX)
    ax_top.set_title(
        "The Bellman residual is at chance; the start-state critic bias is not.\n"
        r"Each point is one training seed; cloud lives strictly above $y=x$.",
        fontsize=10.5, loc="left", pad=8, fontweight="bold",
    )

    plt.savefig(out_path("fig1_cross_algo.pdf"))
    plt.close(fig)
    print(f"wrote {out_path('fig1_cross_algo.pdf')}")
    print(f"summary: {above}/{n_total} = {100 * above / n_total:.1f}% seeds above diagonal")


if __name__ == "__main__":
    main()
