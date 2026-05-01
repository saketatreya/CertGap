"""Figure 7: per-update points in the (A_k, Δ̂_k) plane on 4 representative envs.

Diagonal A_k = Δ̂_k separates beneficial (below) from harmful (above).
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from figures._common import (
    COLOR_BENEFICIAL,
    COLOR_HARMFUL,
    env_seed_paths,
    load_pickle,
    out_path,
    setup_matplotlib,
)

PANELS = ["LunarLander-v3", "Acrobot-v1", "Hopper-v5", "HalfCheetah-v5"]


def collect(env: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    a_chunks: list[np.ndarray] = []
    d_chunks: list[np.ndarray] = []
    h_chunks: list[np.ndarray] = []
    for path in env_seed_paths(env):
        log = load_pickle(path)["log"]
        a_chunks.append(np.asarray(log["A_k"], dtype=float))
        d_chunks.append(np.asarray(log["delta_hat_k"], dtype=float))
        h_chunks.append(np.asarray(log["harmful_k"], dtype=float))
    if not a_chunks:
        return np.empty(0), np.empty(0), np.empty(0)
    return np.concatenate(a_chunks), np.concatenate(d_chunks), np.concatenate(h_chunks)


def main() -> None:
    setup_matplotlib()
    fig, axes = plt.subplots(1, len(PANELS), figsize=(11.5, 3.0), sharex=False, sharey=False)
    for ax, env in zip(axes, PANELS):
        a, d, h = collect(env)
        if a.size == 0:
            ax.set_title(f"{env} (no data)")
            ax.axis("off")
            continue
        mask = np.isfinite(a) & np.isfinite(d) & np.isfinite(h)
        a = a[mask]; d = d[mask]; h = h[mask]
        beneficial = h < 0.5
        ax.scatter(a[~beneficial], d[~beneficial], s=4, alpha=0.4, color=COLOR_HARMFUL,    label="harmful")
        ax.scatter(a[ beneficial], d[ beneficial], s=4, alpha=0.4, color=COLOR_BENEFICIAL, label="beneficial")
        lo = float(min(np.percentile(a, 1), np.percentile(d, 1)))
        hi = float(max(np.percentile(a, 99), np.percentile(d, 99)))
        ax.plot([lo, hi], [lo, hi], "k--", linewidth=0.7)
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_xlabel(r"$A_k$"); ax.set_ylabel(r"$\hat\Delta_k$")
        ax.set_title(env.split("-")[0])
    axes[0].legend(loc="lower right", frameon=False, fontsize=7)
    plt.tight_layout()
    plt.savefig(out_path("appendix/figA2_decomposition.pdf"))
    plt.close(fig)
    print(f"wrote {out_path('appendix/figA2_decomposition.pdf')}")


if __name__ == "__main__":
    main()
