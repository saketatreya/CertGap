"""Figure A3: ε_u-definition robustness on Humanoid-v5.

Three definitions of the Bellman residual scalar — mean-squared (MSE),
sup-norm, and 95th-percentile of |residual| — all sit at chance for
predicting per-update harm on Humanoid. The failure of −ε_u as a harm
predictor is intrinsic to the metric, not to the choice of scalarization.

Reads `results/eps_u_variants/Humanoid-v5/seed_*.pkl`, produced by
`certgap.runners.sweep --grid-name eps_u_variants`.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from certgap.analysis.harm_prediction import auroc
from figures._common import COLOR_EPS_U, load_pickle, out_path, setup_matplotlib

KINDS = [
    ("eps_u_mse", "MSE"),
    ("eps_u_sup", "sup-norm"),
    ("eps_u_p95", r"$95^\mathrm{th}$ pct"),
]


def per_seed_auroc(path: Path, key: str) -> float:
    log = load_pickle(path)["log"]
    if key not in log:
        raise KeyError(f"{path} missing key {key!r}; rerun with --log-eps-u-variants.")
    metric = -np.asarray(log[key], dtype=float)  # high ε_u → predicts harm
    harm = np.asarray(log["harmful_k"], dtype=float)
    return auroc(metric, harm)


def main() -> None:
    setup_matplotlib()
    folder = Path("results/eps_u_variants/Humanoid-v5")
    pkls = sorted(folder.glob("seed_*.pkl"))
    if not pkls:
        raise FileNotFoundError(
            f"No pickles under {folder}. Run "
            f"`python -m certgap.runners.sweep --grid-name eps_u_variants` first."
        )

    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    width = 0.55
    summary: dict[str, dict[str, float]] = {}
    rng = np.random.default_rng(0)

    for i, (key, label) in enumerate(KINDS):
        seed_aurocs = np.asarray([per_seed_auroc(p, key) for p in pkls])
        med = float(np.median(seed_aurocs))
        lo = float(np.percentile(seed_aurocs, 2.5))
        hi = float(np.percentile(seed_aurocs, 97.5))

        ax.bar(i, med, width=width, color=COLOR_EPS_U, alpha=0.55,
               edgecolor="k", linewidth=0.8)
        ax.errorbar(i, med, yerr=[[med - lo], [hi - med]],
                    color="k", capsize=3, linewidth=0.7)
        # Per-seed dots, jittered horizontally
        x_jitter = i + rng.uniform(-0.10, 0.10, size=seed_aurocs.size)
        ax.scatter(x_jitter, seed_aurocs, color="k", s=14, zorder=3, alpha=0.75)

        summary[label] = {
            "median": med, "lo_2.5": lo, "hi_97.5": hi,
            "n_seeds": int(seed_aurocs.size),
            "per_seed": seed_aurocs.tolist(),
        }

    ax.axhline(0.5, color="0.4", linestyle="--", linewidth=0.8, label="chance")
    ax.set_xticks(range(len(KINDS)))
    ax.set_xticklabels([lbl for _, lbl in KINDS])
    ax.set_xlabel(r"$\varepsilon_u$ definition")
    ax.set_ylabel(r"AUROC for harm prediction ($-\varepsilon_u$ vs harm)")
    ax.set_ylim(0.30, 0.70)
    ax.set_title(r"Humanoid-v5: $-\varepsilon_u$ fails under every definition")
    ax.legend(loc="upper right", frameon=False, fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path("appendix/figA3_eps_u_variants.pdf"))
    plt.close(fig)

    summary_path = out_path("appendix/figA3_eps_u_variants.json")
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"wrote {out_path('appendix/figA3_eps_u_variants.pdf')}")
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
