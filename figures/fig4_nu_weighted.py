"""Fig 4 — ν-weighted critic loss evaluation.

Three-panel figure comparing vanilla PPO vs PPO with the ν-weighted critic loss:
  Panel A: |Δ̂_k| over training (mean ± std band across seeds)
  Panel B: Per-env lag-0 and lag-1 AUROC of -Δ̂_k for harm prediction
  Panel C: Final-return distribution per env (paired strip/box)
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from figures._common import (
    setup_matplotlib,
    load_pickle,
    out_path,
    env_seed_paths,
    COLOR_CG,
    COLOR_EPS_U,
    COLOR_NDHAT,
)
from certgap.analysis.harm_prediction import auroc

setup_matplotlib()

ENVS = ["Humanoid-v5", "HalfCheetah-v5", "Hopper-v5"]
ENV_SHORT = {"Humanoid-v5": "Humanoid", "HalfCheetah-v5": "HalfCheetah", "Hopper-v5": "Hopper"}
VANILLA_DIR = "results/main"
NU_DIR = "results/ppo_nu_weighted"

COLOR_VANILLA = "#7f7f7f"
COLOR_NU = "#1f77b4"


def _load_arrays(results_dir, env, key):
    """Load a per-seed array of a log key from results_dir/env/seed_*.pkl."""
    paths = env_seed_paths(env, results_dir)
    arrays = []
    for p in paths:
        data = load_pickle(p)
        log = data["log"]
        if key == "delta_hat_abs" and key not in log:
            arr = np.abs(np.asarray(log["delta_hat_k"], dtype=float))
        else:
            arr = np.asarray(log[key], dtype=float)
        arrays.append(arr)
    return arrays


def _compute_auroc(results_dir, env, lag=0):
    """Compute per-seed AUROC of -delta_hat_k for harm prediction."""
    paths = env_seed_paths(env, results_dir)
    vals = []
    for p in paths:
        data = load_pickle(p)
        log = data["log"]
        m = -np.asarray(log["delta_hat_k"], dtype=float)
        h = np.asarray(log["harmful_k"], dtype=float)
        if lag == 1 and m.size >= 2:
            m, h = m[:-1], h[1:]
        val = auroc(m, h)
        if np.isfinite(val):
            vals.append(val)
    return vals


def _final_returns(results_dir, env):
    """Load the final J_k value from each seed."""
    paths = env_seed_paths(env, results_dir)
    rets = []
    for p in paths:
        data = load_pickle(p)
        rets.append(float(data["log"]["J_k"][-1]))
    return rets


def main():
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.5))

    # --- Panel A: |Δ̂_k| over training ---
    ax = axes[0]
    env = "Humanoid-v5"

    vanilla_dh = _load_arrays(VANILLA_DIR, env, "delta_hat_abs")
    nu_dh = _load_arrays(NU_DIR, env, "delta_hat_abs")

    if vanilla_dh and nu_dh:
        # Truncate to shortest seed length
        min_len_v = min(len(a) for a in vanilla_dh)
        min_len_n = min(len(a) for a in nu_dh)
        v_mat = np.array([a[:min_len_v] for a in vanilla_dh])
        n_mat = np.array([a[:min_len_n] for a in nu_dh])

        v_mean, v_std = v_mat.mean(0), v_mat.std(0)
        n_mean, n_std = n_mat.mean(0), n_mat.std(0)

        x_v = np.arange(min_len_v)
        x_n = np.arange(min_len_n)
        ax.plot(x_v, v_mean, color=COLOR_VANILLA, label="Vanilla PPO", linewidth=1.5)
        ax.fill_between(x_v, v_mean - v_std, v_mean + v_std, color=COLOR_VANILLA, alpha=0.2)
        ax.plot(x_n, n_mean, color=COLOR_NU, label=r"PPO + $\nu$-loss", linewidth=1.5)
        ax.fill_between(x_n, n_mean - n_std, n_mean + n_std, color=COLOR_NU, alpha=0.2)

        reduction = 100 * (1 - n_mean[-1] / v_mean[-1]) if v_mean[-1] > 0 else 0
        ax.set_title(rf"$|\hat\Delta_k|$ on {ENV_SHORT[env]} ($-${reduction:.0f}%)")
    else:
        ax.set_title(rf"$|\hat\Delta_k|$ on {ENV_SHORT[env]} (data pending)")

    ax.set_xlabel("Update")
    ax.set_ylabel(r"$|\hat\Delta_k|$")
    ax.legend(fontsize=7)

    # --- Panel B: Lag-0 and Lag-1 AUROC paired bars ---
    ax = axes[1]
    x = np.arange(len(ENVS))
    width = 0.18
    offsets = [-1.5, -0.5, 0.5, 1.5]
    labels = ["Vanilla lag-0", r"$\nu$-loss lag-0", "Vanilla lag-1", r"$\nu$-loss lag-1"]
    colors = [COLOR_VANILLA, COLOR_NU, COLOR_VANILLA, COLOR_NU]
    hatches = ["", "", "///", "///"]

    all_bars = []
    for env_i, env in enumerate(ENVS):
        v0 = _compute_auroc(VANILLA_DIR, env, lag=0)
        n0 = _compute_auroc(NU_DIR, env, lag=0)
        v1 = _compute_auroc(VANILLA_DIR, env, lag=1)
        n1 = _compute_auroc(NU_DIR, env, lag=1)
        all_bars.append([
            np.median(v0) if v0 else 0.5,
            np.median(n0) if n0 else 0.5,
            np.median(v1) if v1 else 0.5,
            np.median(n1) if n1 else 0.5,
        ])

    bars_arr = np.array(all_bars)
    for j in range(4):
        bars = ax.bar(x + offsets[j] * width, bars_arr[:, j], width,
                      color=colors[j], hatch=hatches[j], edgecolor="white",
                      label=labels[j], alpha=0.85)

    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.7, alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([ENV_SHORT[e] for e in ENVS])
    ax.set_ylabel("AUROC")
    ax.set_title("Harm-prediction AUROC")
    ax.legend(fontsize=6, ncol=2, loc="upper left")
    ax.set_ylim(0.35, 0.85)

    # --- Panel C: Final return distribution ---
    ax = axes[2]
    for i, env in enumerate(ENVS):
        v_ret = _final_returns(VANILLA_DIR, env)
        n_ret = _final_returns(NU_DIR, env)
        if v_ret:
            ax.scatter([i - 0.15] * len(v_ret), v_ret, color=COLOR_VANILLA,
                       alpha=0.7, s=25, zorder=3, label="Vanilla" if i == 0 else "")
            ax.plot([i - 0.25, i - 0.05], [np.mean(v_ret)] * 2,
                    color=COLOR_VANILLA, linewidth=2, zorder=4)
        if n_ret:
            ax.scatter([i + 0.15] * len(n_ret), n_ret, color=COLOR_NU,
                       alpha=0.7, s=25, zorder=3, label=r"$\nu$-loss" if i == 0 else "")
            ax.plot([i + 0.05, i + 0.25], [np.mean(n_ret)] * 2,
                    color=COLOR_NU, linewidth=2, zorder=4)

    ax.set_xticks(range(len(ENVS)))
    ax.set_xticklabels([ENV_SHORT[e] for e in ENVS])
    ax.set_ylabel("Final Return")
    ax.set_title("Final return (parity check)")
    ax.legend(fontsize=7)

    plt.tight_layout()
    dest = out_path("fig4_nu_weighted.pdf")
    plt.savefig(dest)
    print(f"wrote {dest}")
    plt.close()


if __name__ == "__main__":
    main()
