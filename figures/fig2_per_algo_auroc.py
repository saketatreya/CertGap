"""Figure 2: per-algorithm × per-environment AUROC bars.

For each (algorithm, environment) cell we compute per-seed AUROC of
(a) -ε_u and (b) -Δ̂_k for predicting harm. Median of seeds with bootstrap
95% CI, per-seed dots overlaid. Chance line at 0.5. The pattern: gray bars
hover at chance; blue bars sit well above.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from certgap.analysis.harm_prediction import auroc
from figures._common import out_path, setup_matplotlib


# Per-algorithm column: (algo label, results dir, env list)
ALGOS = [
    ("PPO ($n{=}20$/env)",
     "main",
     ["LunarLander-v3", "CartPole-v1", "Acrobot-v1",
      "Hopper-v5", "HalfCheetah-v5", "Walker2d-v5", "Ant-v5", "Humanoid-v5"]),
    ("PPO-MSE ($n{=}5$/env)",
     "ppo_mse",
     ["LunarLander-v3", "Hopper-v5", "HalfCheetah-v5", "Walker2d-v5", "Ant-v5", "Humanoid-v5"]),
    ("SAC ($n{=}10$/env)",
     "sac",
     ["Hopper-v5", "HalfCheetah-v5", "Walker2d-v5"]),
    ("TRPO ($n{=}10$/env)",
     "trpo",
     ["LunarLander-v3", "Hopper-v5"]),
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


def per_seed_auroc(seeds: list[dict], key: str, sign: int) -> np.ndarray:
    out = []
    for d in seeds:
        log = d["log"]
        m = sign * np.asarray(log[key], dtype=float)
        h = np.asarray(log["harmful_k"], dtype=float)
        out.append(auroc(m, h))
    return np.asarray(out, dtype=float)


def bootstrap_ci(x: np.ndarray, n_boot: int = 2000, rng=None) -> tuple[float, float, float]:
    rng = rng or np.random.default_rng(20260501)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan"), float("nan"), float("nan")
    boot = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, x.size, size=x.size)
        boot[b] = np.median(x[idx])
    return float(np.median(x)), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def main() -> None:
    setup_matplotlib()

    n_cols = len(ALGOS)
    fig, axes = plt.subplots(
        1, n_cols, figsize=(13, 3.4),
        gridspec_kw={"width_ratios": [len(envs) for _, _, envs in ALGOS]},
    )

    rng = np.random.default_rng(0)
    width = 0.36
    color_eps = "#7f7f7f"
    color_dh = "#1f77b4"

    for ax, (algo_label, top, envs) in zip(axes, ALGOS):
        positions = np.arange(len(envs))
        eps_meds, eps_los, eps_his = [], [], []
        dh_meds, dh_los, dh_his = [], [], []

        for env in envs:
            seeds = load_seeds(Path(f"results/{top}/{env}"))
            if not seeds:
                eps_meds.append(np.nan); eps_los.append(np.nan); eps_his.append(np.nan)
                dh_meds.append(np.nan); dh_los.append(np.nan); dh_his.append(np.nan)
                continue
            eps_arr = per_seed_auroc(seeds, "eps_u", -1)
            dh_arr = per_seed_auroc(seeds, "delta_hat_k", -1)
            em, el, eh = bootstrap_ci(eps_arr, rng=rng)
            dm, dl, dh = bootstrap_ci(dh_arr, rng=rng)
            eps_meds.append(em); eps_los.append(em - el); eps_his.append(eh - em)
            dh_meds.append(dm); dh_los.append(dm - dl); dh_his.append(dh - dm)

            # per-seed dots, jittered
            x_eps = positions[envs.index(env)] - width / 2
            x_dh = positions[envs.index(env)] + width / 2
            jx_e = rng.uniform(-0.06, 0.06, size=eps_arr.size)
            jx_d = rng.uniform(-0.06, 0.06, size=dh_arr.size)
            ax.scatter(x_eps + jx_e, eps_arr, color="k", s=7, alpha=0.55, zorder=3, linewidths=0)
            ax.scatter(x_dh + jx_d, dh_arr, color="k", s=7, alpha=0.55, zorder=3, linewidths=0)

        ax.bar(positions - width / 2, eps_meds, width=width,
               yerr=[eps_los, eps_his], capsize=2, color=color_eps,
               edgecolor="black", linewidth=0.6, error_kw={"linewidth": 0.7},
               label=r"$-\varepsilon_u$" if ax is axes[0] else None)
        ax.bar(positions + width / 2, dh_meds, width=width,
               yerr=[dh_los, dh_his], capsize=2, color=color_dh,
               edgecolor="black", linewidth=0.6, error_kw={"linewidth": 0.7},
               label=r"$-\hat\Delta_k$" if ax is axes[0] else None)

        ax.axhline(0.5, color="0.5", linestyle="--", linewidth=0.8)
        ax.set_xticks(positions)
        ax.set_xticklabels([e.split("-")[0] for e in envs], rotation=20, ha="right", fontsize=8)
        ax.set_ylim(0.18, 0.92)
        ax.set_title(algo_label, fontsize=10)
        if ax is axes[0]:
            ax.set_ylabel("AUROC for harm prediction", fontsize=10)
            ax.legend(loc="lower right", frameon=False, fontsize=9)

    plt.suptitle(
        "$-\\varepsilon_u$ is at chance on every (algorithm, environment) cell; $-\\hat\\Delta_k$ is well above.",
        fontsize=11, y=1.02,
    )
    plt.tight_layout()
    plt.savefig(out_path("fig2_per_algo_auroc.pdf"))
    plt.close(fig)
    print(f"wrote {out_path('fig2_per_algo_auroc.pdf')}")


if __name__ == "__main__":
    main()
