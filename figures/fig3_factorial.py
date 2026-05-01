"""Figure 4: The Robustness (Factorial Heatmaps).

Layout: 2 rows × 3 columns.
  Row 1 (cert-gap concurrent AUROC): LunarLander | HalfCheetah | Hopper
  Row 2 (ε_u concurrent AUROC): same three envs.

Redesigned with clean Seaborn heatmaps, proper diverging/sequential palettes,
and non-overlapping colorbars.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from certgap.analysis.factorial import assemble_grid
from figures._common import out_path, setup_matplotlib

def gather(folder: Path) -> dict[tuple, list[Path]]:
    cells: dict[tuple, list[Path]] = defaultdict(list)
    if not folder.exists():
        return cells
    for cell_dir in sorted(folder.iterdir()):
        if not cell_dir.is_dir() or not cell_dir.name.startswith("ve"):
            continue
        try:
            ve_part, ce_part = cell_dir.name.split("_", 1)
            ve = int(ve_part[2:])
            ce = float(ce_part[2:])
        except (ValueError, IndexError):
            continue
        for pkl in sorted(cell_dir.glob("seed_*.pkl")):
            cells[(ve, ce)].append(pkl)
    return cells

def render_panel(ax, cells, env_label: str, metric_key: str, vmin, vmax, cmap, cbar_ax, top_row: bool, leftmost: bool):
    if not cells:
        ax.set_title(f"{env_label}\n(no data)", fontsize=10, fontweight="bold")
        ax.axis("off")
        return None
        
    rows, cols, mat = assemble_grid(cells, metric_key, lag=0)
    
    # We want to flip the y-axis so the smallest value-epochs are at the bottom, or just leave as is.
    # Usually heatmaps put the first row at the top. Let's stick to standard.
    
    title = env_label if top_row else None
    
    sns.heatmap(
        mat, 
        ax=ax, 
        vmin=vmin, 
        vmax=vmax, 
        cmap=cmap, 
        annot=True, 
        fmt=".2f", 
        cbar=cbar_ax is not None,
        cbar_ax=cbar_ax,
        annot_kws={"size": 8, "fontweight": "bold"},
        linewidths=0.5,
        linecolor='white'
    )
    
    ax.set_xticks(np.arange(len(cols)) + 0.5)
    ax.set_xticklabels([f"{c:g}" for c in cols], fontsize=8)
    ax.set_yticks(np.arange(len(rows)) + 0.5)
    ax.set_yticklabels([str(r) for r in rows], fontsize=8, rotation=0)
    
    if not top_row:
        ax.set_xlabel(r"Clip $\varepsilon$", fontsize=9, fontweight="bold")
    else:
        ax.set_xlabel("")
        ax.set_xticklabels([])
        
    if leftmost:
        ax.set_ylabel("Value Epochs", fontsize=9, fontweight="bold")
    else:
        ax.set_ylabel("")
        
    if title:
        ax.set_title(title, fontsize=10, fontweight="bold", pad=10)

def main() -> None:
    setup_matplotlib()

    envs = [
        ("LunarLander",  Path("results/factorial_lunarlander")),
        ("HalfCheetah",  Path("results/factorial_halfcheetah")),
        ("Hopper",       Path("results/factorial_hopper")),
    ]
    cells_by_env = {label: gather(folder) for label, folder in envs}

    fig = plt.figure(figsize=(12, 6))
    
    # Create a custom GridSpec to handle colorbars cleanly without overlap
    import matplotlib.gridspec as gridspec
    gs = gridspec.GridSpec(2, 4, width_ratios=[1, 1, 1, 0.05], wspace=0.15, hspace=0.15)
    
    cbar_ax1 = fig.add_subplot(gs[0, 3])
    cbar_ax2 = fig.add_subplot(gs[1, 3])

    # Row 1: cert-gap (Mako palette)
    for j, (env_label, _) in enumerate(envs):
        ax = fig.add_subplot(gs[0, j])
        c_ax = cbar_ax1 if j == 2 else None
        render_panel(
            ax, cells_by_env[env_label], env_label,
            metric_key="cert_gap_k", vmin=0.5, vmax=1.0, cmap="mako_r",
            cbar_ax=c_ax, top_row=True, leftmost=(j == 0)
        )

    # Row 2: ε_u (Rocket palette)
    for j, (env_label, _) in enumerate(envs):
        ax = fig.add_subplot(gs[1, j])
        c_ax = cbar_ax2 if j == 2 else None
        render_panel(
            ax, cells_by_env[env_label], env_label,
            metric_key="eps_u", vmin=0.3, vmax=0.65, cmap="rocket_r",
            cbar_ax=c_ax, top_row=False, leftmost=(j == 0)
        )

    # Row labels (left-side margin annotations)
    fig.text(0.01, 0.72, r"CertGap AUROC", rotation=90, va="center", ha="left", fontsize=11, fontweight="bold", color="#2c7fb8")
    fig.text(0.01, 0.28, r"$\varepsilon_u$ AUROC", rotation=90, va="center", ha="left", fontsize=11, fontweight="bold", color="#e6550d")

    plt.tight_layout(rect=[0.03, 0, 0.98, 1])
    plt.savefig(out_path("fig3_factorial.pdf"))
    plt.close(fig)
    print(f"wrote {out_path('fig3_factorial.pdf')}")

if __name__ == "__main__":
    main()
