"""Figure A1: Per-Seed Pairwise Comparison (Dumbbell Plot).

Plots a paired dumbbell for every single seed (160 runs total), 
showing the shift in AUROC from eps_u (gray) to cert_gap (blue).
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from certgap.analysis.harm_prediction import auroc
from figures._common import (
    COLOR_CG,
    COLOR_EPS_U,
    ENV_ORDER,
    env_seed_paths,
    load_pickle,
    out_path,
    setup_matplotlib,
)

def per_seed_auroc(paths, metric_key: str, sign: int) -> list[float]:
    out = []
    for p in paths:
        log = load_pickle(p)["log"]
        m = np.asarray(log[metric_key], dtype=float) * sign
        h = np.asarray(log["harmful_k"], dtype=float)
        out.append(auroc(m, h))
    return out

def main() -> None:
    setup_matplotlib()
    
    rows = []
    for env in ENV_ORDER:
        paths = env_seed_paths(env)
        if not paths: continue
        env_name = env.split("-")[0]
        
        eps_u_aurocs = per_seed_auroc(paths, "eps_u", -1)
        cg_aurocs = per_seed_auroc(paths, "cert_gap_k", +1)
        
        for i, (eps, cg) in enumerate(zip(eps_u_aurocs, cg_aurocs)):
            if np.isfinite(eps) and np.isfinite(cg):
                rows.append({
                    "Environment": env_name, 
                    "Seed": i, 
                    "eps_u": eps, 
                    "cert_gap": cg
                })
                    
    df = pd.DataFrame(rows)
    if df.empty:
        print("No data found.")
        return

    # Sort by eps_u AUROC to make a clean "wall"
    df = df.sort_values(by="eps_u").reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(7, 10))
    
    y = np.arange(len(df))
    
    # Plot lines
    ax.hlines(y=y, xmin=df["eps_u"], xmax=df["cert_gap"], color="black", alpha=0.15, linewidth=1.0, zorder=1)
    
    # Plot dots
    ax.scatter(df["eps_u"], y, color=COLOR_EPS_U, s=15, alpha=0.8, label=r"Bellman Residual $\varepsilon_u$", zorder=2, edgecolors='none')
    ax.scatter(df["cert_gap"], y, color=COLOR_CG, s=15, alpha=0.9, label="CertGap", zorder=3, edgecolors='none')

    ax.axvline(0.5, color="k", linewidth=1.2, linestyle="--", zorder=0)
    
    ax.set_xlabel("Harm Prediction AUROC", fontweight="bold")
    ax.set_ylabel("160 Independent Training Seeds\n(Sorted by $\epsilon_u$ performance)", fontweight="bold")
    ax.set_xlim(0.2, 1.0)
    ax.set_yticks([])
    ax.legend(loc="lower right", frameon=True, fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path("figA1_dumbbell.pdf"))
    plt.close(fig)
    print(f"wrote {out_path('figA1_dumbbell.pdf')}")

if __name__ == "__main__":
    main()
