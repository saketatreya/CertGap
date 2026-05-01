"""Figure 3: The Distribution (Joyplot/Ridgeplot of Conditional Densities).

Bins the actual $\Delta J$ into deciles. Plots the density of the 
metric (eps_u or cert_gap) conditional on the $\Delta J$ decile.
Proves predictive tracking without relying on a linear correlation coefficient.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from figures._common import (
    COLOR_CG,
    COLOR_EPS_U,
    ENV_ORDER,
    env_seed_paths,
    load_pickle,
    out_path,
    setup_matplotlib,
)

def collect_data() -> pd.DataFrame:
    rows = []
    for env in ENV_ORDER:
        for path in env_seed_paths(env):
            data = load_pickle(path)
            log = data["log"]
            eps = np.asarray(log["eps_u"], dtype=float)
            cg = np.asarray(log["cert_gap_k"], dtype=float)
            dj = np.asarray(log["delta_J_k"], dtype=float)
            mask = np.isfinite(eps) & np.isfinite(cg) & np.isfinite(dj)
            eps = eps[mask]
            cg = cg[mask]
            dj = dj[mask]
            if eps.size > 1:
                # Z-score within seed to standardize scale
                eps = (eps - np.mean(eps)) / max(np.std(eps), 1e-9)
                cg = (cg - np.mean(cg)) / max(np.std(cg), 1e-9)
                dj = (dj - np.mean(dj)) / max(np.std(dj), 1e-9)
                for e, c, d in zip(eps, cg, dj):
                    rows.append({"eps_u": e, "cert_gap": c, "delta_j": d})
    return pd.DataFrame(rows)

def main() -> None:
    setup_matplotlib()
    df = collect_data()
    if df.empty:
        print("No data found.")
        return

    # Bin delta_j into 10 quantiles
    df["dj_decile"] = pd.qcut(df["delta_j"], q=10, labels=[f"Q{i+1}" for i in range(10)])
    
    # We want to plot two ridgeplots side by side.
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=True)
    
    deciles = [f"Q{i+1}" for i in range(10)][::-1] # Plot Q10 at top, Q1 at bottom
    
    for ax, metric, color, title in zip(axes, 
                                        ["eps_u", "cert_gap"], 
                                        [COLOR_EPS_U, COLOR_CG],
                                        [r"Bellman Residual $\varepsilon_u$", r"CertGap $A_k - \hat{\Delta}_k$"]):
        
        # We manually build the ridgeplot using fill_between for maximum control
        y_offsets = np.linspace(0, 5, 10) # Spacing
        
        for idx, decile in enumerate(deciles):
            subset = df[df["dj_decile"] == decile][metric]
            
            # Compute KDE
            from scipy.stats import gaussian_kde
            kde = gaussian_kde(subset)
            x_vals = np.linspace(-3, 3, 200)
            y_vals = kde(x_vals)
            
            # Scale y_vals to fit
            y_vals = y_vals / np.max(y_vals) * 0.8
            
            base_y = y_offsets[idx]
            
            ax.plot(x_vals, base_y + y_vals, color="white", linewidth=1.5, zorder=10-idx)
            ax.fill_between(x_vals, base_y, base_y + y_vals, color=color, alpha=0.8, zorder=10-idx, edgecolor='k', linewidth=0.5)
            
            if ax == axes[0]:
                ax.text(-3.2, base_y + 0.1, decile, ha="right", va="center", fontsize=8, fontweight="bold")
                
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Metric Value (z-scored)", fontweight="bold")
        ax.set_xlim(-3, 3)
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.axvline(0, color="k", linestyle="--", alpha=0.3, zorder=0)

    # Add decile explanations
    fig.text(0.02, 0.9, "Highest $\Delta J$\n(Amazing Updates)", ha="left", va="center", fontsize=9, fontweight="bold", color="green")
    fig.text(0.02, 0.1, "Lowest $\Delta J$\n(Disastrous Harm)", ha="left", va="center", fontsize=9, fontweight="bold", color="red")

    plt.tight_layout(rect=[0.05, 0, 1, 1])
    plt.savefig(out_path("fig5_pooled_scatter.pdf"))
    plt.close(fig)
    print(f"wrote {out_path('fig5_pooled_scatter.pdf')}")

if __name__ == "__main__":
    main()
