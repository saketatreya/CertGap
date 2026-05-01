"""Shared helpers for figure scripts.

Every script in `figures/` writes a single PDF to `figures/out/`. The
helpers here load pickles, set matplotlib defaults, and centralize env
ordering / colors so figures are visually consistent.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


# Paper Table 1 environment order. All figures reuse this.
ENV_ORDER = [
    "LunarLander-v3",
    "CartPole-v1",
    "Acrobot-v1",
    "Hopper-v5",
    "HalfCheetah-v5",
    "Walker2d-v5",
    "Ant-v5",
    "Humanoid-v5",
]

# Metric color palette (paper-aligned).
COLOR_CG = "#1f77b4"        # cert-gap → blue
COLOR_EPS_U = "#7f7f7f"     # ε_u → grey
COLOR_A_K = "#9467bd"       # A_k alone → purple
COLOR_NDHAT = "#ff7f0e"     # −Δ̂ alone → orange
COLOR_HARMFUL = "#d62728"   # harmful → red
COLOR_BENEFICIAL = "#2ca02c"  # beneficial → green


def setup_matplotlib() -> None:
    mpl.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def load_pickle(path: str | Path) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def env_seed_paths(env: str, results_dir: str | Path = "results/main") -> list[Path]:
    """Sorted seed-pickle paths for an env under `results_dir/env/`."""
    base = Path(results_dir) / env
    if not base.exists():
        return []
    return sorted(base.glob("seed_*.pkl"))


def out_path(filename: str) -> Path:
    """Resolve a path under figures/out/. Filename may include subdirs
    (e.g. 'appendix/figA1.pdf'); intermediate directories are created."""
    out = Path(__file__).resolve().parent / "out"
    target = out / filename
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


def pool_metric_harm(
    pkl_paths: list[Path], metric_key: str, *, lag: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Concatenate (metric, harm) arrays across seeds, optionally lag-shifted."""
    metric_chunks: list[np.ndarray] = []
    harm_chunks: list[np.ndarray] = []
    for pkl in pkl_paths:
        data = load_pickle(pkl)
        log = data["log"]
        m = np.asarray(log[metric_key], dtype=float)
        h = np.asarray(log["harmful_k"], dtype=float)
        if lag == 0:
            metric_chunks.append(m)
            harm_chunks.append(h)
        elif lag == 1:
            if m.size < 2:
                continue
            metric_chunks.append(m[:-1])
            harm_chunks.append(h[1:])
        else:
            raise ValueError(f"Unsupported lag={lag}")
    if not metric_chunks:
        return np.empty(0), np.empty(0)
    return np.concatenate(metric_chunks), np.concatenate(harm_chunks)


def per_seed_arrays(pkl_paths: list[Path], metric_key: str) -> list[tuple[np.ndarray, np.ndarray]]:
    out = []
    for pkl in pkl_paths:
        data = load_pickle(pkl)
        log = data["log"]
        out.append((
            np.asarray(log[metric_key], dtype=float),
            np.asarray(log["harmful_k"], dtype=float),
        ))
    return out
