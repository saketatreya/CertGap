"""Pearson, Spearman, paired Wilcoxon — paper Section 4 / Appendix B."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass
class Correlation:
    r: float
    lo: float  # 95% CI lower (Fisher z)
    hi: float  # 95% CI upper (Fisher z)
    p: float
    n: int


def _fisher_z_ci(r: float, n: int) -> tuple[float, float]:
    if n < 4 or not np.isfinite(r) or abs(r) >= 1.0:
        return float("nan"), float("nan")
    z = np.arctanh(r)
    se = 1.0 / np.sqrt(n - 3)
    return float(np.tanh(z - 1.96 * se)), float(np.tanh(z + 1.96 * se))


def pearson(x: np.ndarray, y: np.ndarray) -> Correlation:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size < 3:
        return Correlation(float("nan"), float("nan"), float("nan"), float("nan"), x.size)
    res = stats.pearsonr(x, y)
    lo, hi = _fisher_z_ci(float(res.statistic), x.size)
    return Correlation(float(res.statistic), lo, hi, float(res.pvalue), x.size)


def spearman(x: np.ndarray, y: np.ndarray) -> Correlation:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size < 3:
        return Correlation(float("nan"), float("nan"), float("nan"), float("nan"), x.size)
    res = stats.spearmanr(x, y)
    lo, hi = _fisher_z_ci(float(res.statistic), x.size)
    return Correlation(float(res.statistic), lo, hi, float(res.pvalue), x.size)


def paired_wilcoxon_one_sided_greater(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """One-sided Wilcoxon signed-rank: H1 is `a > b` (per-pair).

    Returns (statistic, p-value). Used in paper Table 1 to compare
    cert-gap AUROC vs ε_u AUROC across paired seeds.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    a = a[mask]
    b = b[mask]
    if a.size < 5:
        # paper reports `—` for n < 5
        return float("nan"), float("nan")
    res = stats.wilcoxon(a, b, alternative="greater", zero_method="wilcox")
    return float(res.statistic), float(res.pvalue)


def harm_base_rate(harm: np.ndarray) -> float:
    harm = np.asarray(harm, dtype=float)
    mask = np.isfinite(harm)
    if mask.sum() == 0:
        return float("nan")
    return float(harm[mask].mean())
