"""Hyperparameter-factorial heatmap aggregation (Figure 3)."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

from certgap.analysis.harm_prediction import auroc


def cell_auroc(pkl_paths: list[Path], metric_key: str) -> float:
    """Pool per-update arrays across seeds in this cell, return AUROC."""
    metric_chunks: list[np.ndarray] = []
    harm_chunks: list[np.ndarray] = []
    for pkl in pkl_paths:
        with open(pkl, "rb") as f:
            data = pickle.load(f)
        log = data["log"]
        metric_chunks.append(np.asarray(log[metric_key], dtype=float))
        harm_chunks.append(np.asarray(log["harmful_k"], dtype=float))
    metric = np.concatenate(metric_chunks)
    harm = np.concatenate(harm_chunks)
    return auroc(metric, harm)


def cell_lag1_auroc(pkl_paths: list[Path], metric_key: str) -> float:
    """Lag-1 AUROC: metric at update k, harm at update k+1.

    Section 6 of the paper. Concurrent AUROC compares update-k metric vs
    update-k harm; lag-1 AUROC shifts the harm by one. The paper shows
    lag-1 collapses to chance even when concurrent is 0.83.
    """
    metric_chunks: list[np.ndarray] = []
    harm_chunks: list[np.ndarray] = []
    for pkl in pkl_paths:
        with open(pkl, "rb") as f:
            data = pickle.load(f)
        log = data["log"]
        metric = np.asarray(log[metric_key], dtype=float)
        harm = np.asarray(log["harmful_k"], dtype=float)
        if metric.size < 2:
            continue
        metric_chunks.append(metric[:-1])
        harm_chunks.append(harm[1:])
    if not metric_chunks:
        return float("nan")
    return auroc(np.concatenate(metric_chunks), np.concatenate(harm_chunks))


def assemble_grid(
    cells: dict[tuple, list[Path]],
    metric_key: str,
    *,
    lag: int = 0,
) -> tuple[list, list, np.ndarray]:
    """Given {cell_key: [pkl, ...]}, return (row_labels, col_labels, AUROC matrix).

    `cell_key` is a 2-tuple (row, col), e.g. (value_epochs, clip_eps).
    """
    rows = sorted({k[0] for k in cells})
    cols = sorted({k[1] for k in cells})
    matrix = np.full((len(rows), len(cols)), np.nan)
    fn = cell_lag1_auroc if lag == 1 else cell_auroc
    for (r, c), paths in cells.items():
        i = rows.index(r)
        j = cols.index(c)
        matrix[i, j] = fn(paths, metric_key)
    return rows, cols, matrix
