"""Figure 9: per-update certification gap and actual ΔJ on LunarLander seed 0.

Pure exemplar — shows that cg_k tracks ΔJ_k visually, not as a one-step
controller but as a calibrated post-hoc diagnostic.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from figures._common import COLOR_CG, load_pickle, out_path, setup_matplotlib


def main() -> None:
    setup_matplotlib()
    path = Path("results/main/LunarLander-v3/seed_0.pkl")
    if not path.exists():
        raise FileNotFoundError(f"{path} missing — run the main grid first.")
    log = load_pickle(path)["log"]
    cg = np.asarray(log["cert_gap_k"], dtype=float)
    dj = np.asarray(log["delta_J_k"], dtype=float)

    fig, ax = plt.subplots(figsize=(8.5, 3.0))
    ax.plot(dj, color="0.3", linewidth=0.8, label=r"actual $\Delta J_k$")
    ax.plot(cg, color=COLOR_CG, linewidth=1.0, alpha=0.95,
            label=r"cert-gap $A_k - \hat\Delta_k$")
    # Fill between to show co-movement.
    ax.fill_between(np.arange(cg.size), 0, np.minimum(cg, dj),
                    where=np.minimum(cg, dj) < 0, color="red", alpha=0.15, linewidth=0)
    ax.axhline(0, color="0.7", linewidth=0.5)
    ax.set_xlabel("PPO update k")
    ax.set_ylabel(r"$\Delta J$ / cert-gap")
    ax.set_title("LunarLander seed 0: cert-gap tracks actual $\\Delta J$")
    ax.legend(loc="upper right", frameon=False, fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path("appendix/figA5_trajectory.pdf"))
    plt.close(fig)
    print(f"wrote {out_path('appendix/figA5_trajectory.pdf')}")


if __name__ == "__main__":
    main()
