"""Figure 5: tabular identity histogram (Appendix A).

Reads `results/tabular/identity_residuals.json` (a list of |A_k - Δ̂_k - ΔJ_k|
samples across MDP families × seeds × NPG iterations) and plots the
log-scale histogram. Median should be O(10⁻¹⁵), max O(10⁻¹⁴).
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from figures._common import out_path, setup_matplotlib


def main() -> None:
    setup_matplotlib()
    path = Path("results/tabular/identity_residuals.json")
    if not path.exists():
        raise FileNotFoundError(
            "results/tabular/identity_residuals.json not found. "
            "Run `python scripts/run_all.py` first."
        )
    data = json.loads(path.read_text())  # {family_label: [residuals]}
    fig, ax = plt.subplots(figsize=(7, 3.4))
    bins = np.logspace(-17, -13, 41)
    colors = plt.get_cmap("tab10")
    all_vals: list[float] = []
    for i, (label, vals) in enumerate(data.items()):
        ax.hist(vals, bins=bins, alpha=0.55, label=label, color=colors(i % 10))
        all_vals.extend(vals)
    median = float(np.median(all_vals))
    maximum = float(np.max(all_vals))
    ax.axvline(np.finfo(float).eps, linestyle="--", color="k", linewidth=0.8, label="machine $\\varepsilon$")
    ax.set_xscale("log")
    ax.set_xlabel(r"$|(A_k - \hat\Delta_k) - (J(\pi_{k+1}) - J(\pi_k))|$")
    ax.set_ylabel("count")
    ax.set_title(f"tabular identity residual: median {median:.1e}, max {maximum:.1e}")
    ax.legend(loc="upper right", frameon=False, fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path("appendix/figA1_tabular_identity.pdf"))
    plt.close(fig)
    print(f"wrote {out_path('appendix/figA1_tabular_identity.pdf')}")


if __name__ == "__main__":
    main()
