"""Figure 8: pooled scatter of ΔJ_{k+1} vs ΔJ_k on 4 representative envs.

Three of four exhibit clear negative lag-1 autocorrelation, an empirical
effect that pushes lag-1 AUROC below chance in deep RL but does not
drive the structural collapse to chance (which holds in tabular too).
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from certgap.analysis.correlations import pearson
from figures._common import (
    env_seed_paths,
    load_pickle,
    out_path,
    setup_matplotlib,
)

PANELS = ["LunarLander-v3", "Acrobot-v1", "Hopper-v5", "HalfCheetah-v5"]


def collect(env: str) -> tuple[np.ndarray, np.ndarray]:
    pairs_x: list[np.ndarray] = []
    pairs_y: list[np.ndarray] = []
    for path in env_seed_paths(env):
        log = load_pickle(path)["log"]
        dj = np.asarray(log["delta_J_k"], dtype=float)
        if dj.size < 2:
            continue
        pairs_x.append(dj[:-1])
        pairs_y.append(dj[1:])
    if not pairs_x:
        return np.empty(0), np.empty(0)
    return np.concatenate(pairs_x), np.concatenate(pairs_y)


def main() -> None:
    setup_matplotlib()
    fig, axes = plt.subplots(1, len(PANELS), figsize=(11.5, 3.0))
    for ax, env in zip(axes, PANELS):
        x, y = collect(env)
        if x.size == 0:
            ax.set_title(f"{env} (no data)")
            ax.axis("off")
            continue
        mask = np.isfinite(x) & np.isfinite(y)
        x = x[mask]; y = y[mask]
        ax.scatter(x, y, s=3, alpha=0.35, color="0.4", linewidths=0)
        if x.size >= 3:
            slope, intercept = np.polyfit(x, y, 1)
            xs = np.linspace(np.percentile(x, 1), np.percentile(x, 99), 100)
            ax.plot(xs, intercept + slope * xs, color="red", linewidth=1.0)
            r = pearson(x, y).r
            ax.text(0.05, 0.95, f"r = {r:+.3f}", transform=ax.transAxes,
                    ha="left", va="top", fontsize=8, color="red")
        ax.axhline(0, color="0.7", linewidth=0.5)
        ax.axvline(0, color="0.7", linewidth=0.5)
        ax.set_xlabel(r"$\Delta J_k$")
        ax.set_ylabel(r"$\Delta J_{k+1}$")
        ax.set_title(env.split("-")[0])
    plt.tight_layout()
    plt.savefig(out_path("appendix/figA4_lag1_autocorrelation.pdf"))
    plt.close(fig)
    print(f"wrote {out_path('appendix/figA4_lag1_autocorrelation.pdf')}")


if __name__ == "__main__":
    main()
