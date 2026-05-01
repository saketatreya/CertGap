"""Harm-prediction analyses: AUROC, AUPRC, calibration, bootstrap CIs.

Every function here operates on per-update arrays produced by `train_ppo`.
A single seed gives one (metric_array, harm_array) pair; a multi-seed
analysis pools the arrays per environment.

Convention: `harm` is a 0/1 array. `metric` is the predictor (cert-gap,
−ε_u, A_k, etc.). Higher metric ⇒ predicts beneficial. NaNs (the final
update has no successor) are filtered before scoring.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

DEFAULT_BOOTSTRAP = 1000
DEFAULT_RNG_SEED = 20260429  # frozen for reproducibility


# --------------------------------------------------------------------- AUROC


def auroc(metric: np.ndarray, harm: np.ndarray) -> float:
    """AUROC for the predictor `metric` against label `(1 - harm)`.

    Higher metric ⇒ beneficial. We score AUROC for predicting the
    *beneficial* class so that `1 − harm` is the positive label, i.e.
    AUROC ≥ 0.5 means the metric is informative in the expected direction.

    Equivalent to AUROC for `−metric` predicting harm (paper convention).
    """
    metric = np.asarray(metric, dtype=float)
    harm = np.asarray(harm, dtype=float)
    mask = np.isfinite(metric) & np.isfinite(harm)
    metric = metric[mask]
    harm = harm[mask]
    if metric.size == 0:
        return float("nan")
    pos = metric[harm < 0.5]
    neg = metric[harm >= 0.5]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    # Mann-Whitney U / (n_pos * n_neg)  with mid-rank for ties.
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    ranks = np.empty(order.size, dtype=float)
    ranks[order] = _avg_ranks(np.concatenate([pos, neg])[order])
    pos_rank_sum = ranks[: pos.size].sum()
    u = pos_rank_sum - pos.size * (pos.size + 1) / 2.0
    return float(u / (pos.size * neg.size))


def _avg_ranks(sorted_values: np.ndarray) -> np.ndarray:
    n = sorted_values.size
    ranks = np.empty(n, dtype=float)
    i = 0
    while i < n:
        j = i + 1
        while j < n and sorted_values[j] == sorted_values[i]:
            j += 1
        avg = (i + 1 + j) / 2.0
        ranks[i:j] = avg
        i = j
    return ranks


def auprc(metric: np.ndarray, harm: np.ndarray) -> float:
    """Area under the precision-recall curve for predicting *harm*.

    Higher metric ⇒ beneficial, so we use −metric as the harm score.
    """
    metric = np.asarray(metric, dtype=float)
    harm = np.asarray(harm, dtype=float).astype(int)
    mask = np.isfinite(metric) & np.isfinite(harm)
    score = -metric[mask]
    label = harm[mask]
    if score.size == 0 or label.sum() == 0:
        return float("nan")
    order = np.argsort(-score, kind="mergesort")
    label = label[order]
    tp = np.cumsum(label)
    fp = np.cumsum(1 - label)
    precision = tp / (tp + fp)
    recall = tp / max(label.sum(), 1)
    # Trapezoidal AUC over recall.
    recall = np.concatenate([[0.0], recall])
    precision = np.concatenate([[1.0], precision])
    return float(np.trapezoid(precision, recall))


# --------------------------------------------------------------------- bootstrap


@dataclass
class BootstrapCI:
    point: float
    lo: float  # 2.5th percentile
    hi: float  # 97.5th percentile

    def as_tuple(self) -> tuple[float, float, float]:
        return self.point, self.lo, self.hi


def bootstrap_metric(
    metric: np.ndarray,
    harm: np.ndarray,
    fn=auroc,
    n_boot: int = DEFAULT_BOOTSTRAP,
    rng: np.random.Generator | None = None,
) -> BootstrapCI:
    """Nonparametric bootstrap CI on `fn(metric, harm)` over PPO updates.

    Resamples with replacement at the per-update granularity. Use this
    *within* a seed (n=#updates per seed) or *across* seeds with the seed
    as the resampling unit.
    """
    rng = rng or np.random.default_rng(DEFAULT_RNG_SEED)
    metric = np.asarray(metric, dtype=float)
    harm = np.asarray(harm, dtype=float)
    mask = np.isfinite(metric) & np.isfinite(harm)
    m = metric[mask]
    h = harm[mask]
    n = m.size
    point = fn(m, h)
    if n == 0:
        return BootstrapCI(float("nan"), float("nan"), float("nan"))
    samples = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        samples[b] = fn(m[idx], h[idx])
    samples = samples[np.isfinite(samples)]
    if samples.size == 0:
        return BootstrapCI(point, float("nan"), float("nan"))
    return BootstrapCI(point, float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5)))


def paired_bootstrap_margin(
    metric_a: np.ndarray,
    metric_b: np.ndarray,
    harm: np.ndarray,
    n_boot: int = DEFAULT_BOOTSTRAP,
    rng: np.random.Generator | None = None,
) -> BootstrapCI:
    """CI on AUROC(a) − AUROC(b) using paired resamples of the same updates.

    The per-update pairing is what makes the comparison powerful — both
    metrics are scored on the same resampled rows.
    """
    rng = rng or np.random.default_rng(DEFAULT_RNG_SEED)
    metric_a = np.asarray(metric_a, dtype=float)
    metric_b = np.asarray(metric_b, dtype=float)
    harm = np.asarray(harm, dtype=float)
    mask = np.isfinite(metric_a) & np.isfinite(metric_b) & np.isfinite(harm)
    a = metric_a[mask]
    b = metric_b[mask]
    h = harm[mask]
    n = h.size
    point = auroc(a, h) - auroc(b, h)
    if n == 0:
        return BootstrapCI(float("nan"), float("nan"), float("nan"))
    samples = np.empty(n_boot, dtype=float)
    for k in range(n_boot):
        idx = rng.integers(0, n, size=n)
        samples[k] = auroc(a[idx], h[idx]) - auroc(b[idx], h[idx])
    samples = samples[np.isfinite(samples)]
    if samples.size == 0:
        return BootstrapCI(point, float("nan"), float("nan"))
    return BootstrapCI(point, float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5)))


# --------------------------------------------------------------------- calibration


def calibration_deciles(
    metric: np.ndarray, harm: np.ndarray, n_bins: int = 10
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (bin_centers, harm_rate_per_bin, count_per_bin).

    Bins are equal-mass quantile bins on `metric` (paper Figure 1 / 6).
    """
    metric = np.asarray(metric, dtype=float)
    harm = np.asarray(harm, dtype=float)
    mask = np.isfinite(metric) & np.isfinite(harm)
    metric = metric[mask]
    harm = harm[mask]
    if metric.size == 0:
        empty = np.full(n_bins, np.nan)
        return empty, empty, np.zeros(n_bins, dtype=int)
    quantiles = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.quantile(metric, quantiles)
    edges[0] -= 1e-12  # ensure leftmost included
    bin_idx = np.clip(np.searchsorted(edges[1:-1], metric), 0, n_bins - 1)
    centers = np.array(
        [np.mean(metric[bin_idx == b]) if (bin_idx == b).any() else np.nan for b in range(n_bins)]
    )
    rates = np.array(
        [np.mean(harm[bin_idx == b]) if (bin_idx == b).any() else np.nan for b in range(n_bins)]
    )
    counts = np.array([(bin_idx == b).sum() for b in range(n_bins)])
    return centers, rates, counts
