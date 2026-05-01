"""Figure 1: per-decile harm rate by metric percentile, 4 representative envs.

Pools updates across seeds for each env, sorts by metric, partitions into 10
equal-mass bins, plots P(ΔJ < 0) per bin.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from certgap.analysis.harm_prediction import calibration_deciles
from figures._common import (
    COLOR_CG,
    COLOR_EPS_U,
    env_seed_paths,
    out_path,
    pool_metric_harm,
    setup_matplotlib,
)

PANELS = ["LunarLander-v3", "Acrobot-v1", "Hopper-v5", "Humanoid-v5"]


def main() -> None:
    setup_matplotlib()
    fig, axes = plt.subplots(1, len(PANELS), figsize=(12, 3), sharey=True)
    for ax, env in zip(axes, PANELS):
        paths = env_seed_paths(env)
        if not paths:
            ax.set_title(f"{env} (no data)")
            continue
        cg, harm = pool_metric_harm(paths, "cert_gap_k")
        eps, _ = pool_metric_harm(paths, "eps_u")

        # Sort cert-gap ascending (low cg = predicted harmful);
        # sort ε_u descending (paper convention: high ε_u = predicted harmful).
        _, cg_rates, _ = calibration_deciles(cg, harm)
        _, eps_rates, _ = calibration_deciles(-eps, harm)

        x = np.arange(1, 11) * 10
        base = float(np.nanmean(harm))

        ax.axhline(base, color="k", linestyle=":", linewidth=1.2, alpha=0.7)
        
        # Reliability diagram style
        ax.plot(x, cg_rates, marker="o", markersize=5, color=COLOR_CG, label="Cert-Gap", linewidth=2.0)
        ax.fill_between(x, cg_rates - 0.05, cg_rates + 0.05, color=COLOR_CG, alpha=0.1)
        
        ax.plot(x, eps_rates, marker="s", markersize=5, color=COLOR_EPS_U, label=r"$\varepsilon_u$", linestyle="--", linewidth=2.0)
        ax.fill_between(x, eps_rates - 0.05, eps_rates + 0.05, color=COLOR_EPS_U, alpha=0.1)
        
        ax.set_title(env.split("-")[0], fontweight="bold")
        ax.set_xlabel("Metric Percentile", fontweight="bold")
        ax.set_ylim(-0.05, 1.05)
        ax.set_xticks([10, 50, 90])
        
        # Make the background a bit styled
        ax.grid(True, axis='y', linestyle='--', alpha=0.3)

    axes[0].set_ylabel(r"Harm Probability $\mathbb{P}[\Delta J<0]$", fontweight="bold")
    axes[0].legend(loc="upper right", frameon=True, fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path("fig2_calibration.pdf"))
    plt.close(fig)
    print(f"wrote {out_path('fig2_calibration.pdf')}")


if __name__ == "__main__":
    main()
