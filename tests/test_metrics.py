"""Unit tests for the AUROC / Bellman-residual / cert-gap math."""

from __future__ import annotations

import numpy as np

from certgap.analysis.correlations import paired_wilcoxon_one_sided_greater, pearson, spearman
from certgap.analysis.harm_prediction import (
    auprc,
    auroc,
    bootstrap_metric,
    calibration_deciles,
    paired_bootstrap_margin,
)


def test_auroc_perfect_predictor() -> None:
    metric = np.array([0.0, 0.1, 0.2, 0.9, 1.0])
    harm   = np.array([1.0, 1.0, 1.0, 0.0, 0.0])
    assert auroc(metric, harm) == 1.0


def test_auroc_worst_predictor() -> None:
    metric = np.array([0.0, 0.1, 0.2, 0.9, 1.0])
    harm   = np.array([0.0, 0.0, 0.0, 1.0, 1.0])
    assert auroc(metric, harm) == 0.0


def test_auroc_chance() -> None:
    rng = np.random.default_rng(0)
    metric = rng.standard_normal(2000)
    harm = rng.binomial(1, 0.4, size=2000).astype(float)
    a = auroc(metric, harm)
    assert 0.45 < a < 0.55


def test_auroc_handles_nan() -> None:
    m = np.array([1.0, np.nan, 0.5, 2.0])
    h = np.array([0.0, 0.0, 1.0, 0.0])
    a = auroc(m, h)
    assert 0.0 <= a <= 1.0


def test_auprc_above_base_rate_when_informative() -> None:
    rng = np.random.default_rng(1)
    n = 1000
    metric = rng.standard_normal(n)
    harm = (metric < -0.3).astype(float)  # strong inverse signal
    base = float(harm.mean())
    assert auprc(metric, harm) > base


def test_bootstrap_ci_brackets_point() -> None:
    rng = np.random.default_rng(2)
    metric = rng.standard_normal(500)
    harm = (metric < 0).astype(float)
    ci = bootstrap_metric(metric, harm, n_boot=200, rng=rng)
    assert ci.lo <= ci.point <= ci.hi


def test_paired_bootstrap_margin_zero_when_metrics_equal() -> None:
    rng = np.random.default_rng(3)
    metric = rng.standard_normal(500)
    harm = (metric < 0).astype(float)
    ci = paired_bootstrap_margin(metric, metric, harm, n_boot=200, rng=rng)
    assert abs(ci.point) < 1e-9
    assert ci.lo <= 0.0 <= ci.hi


def test_calibration_deciles_shape() -> None:
    rng = np.random.default_rng(4)
    metric = rng.standard_normal(1000)
    harm = (metric < 0).astype(float)
    centers, rates, counts = calibration_deciles(metric, harm)
    assert centers.shape == (10,)
    assert rates.shape == (10,)
    assert counts.sum() == 1000
    # monotone-ish: low metric ⇒ harmful
    assert rates[0] > rates[-1]


def test_pearson_matches_numpy() -> None:
    rng = np.random.default_rng(5)
    x = rng.standard_normal(200)
    y = 0.5 * x + 0.5 * rng.standard_normal(200)
    r_ours = pearson(x, y).r
    r_np = float(np.corrcoef(x, y)[0, 1])
    assert abs(r_ours - r_np) < 1e-9


def test_spearman_runs() -> None:
    rng = np.random.default_rng(6)
    x = rng.standard_normal(200)
    y = x ** 3
    r = spearman(x, y).r
    assert r > 0.99


def test_wilcoxon_one_sided() -> None:
    rng = np.random.default_rng(7)
    a = rng.standard_normal(20) + 0.5
    b = rng.standard_normal(20)
    _, p = paired_wilcoxon_one_sided_greater(a, b)
    assert 0.0 < p < 0.05


def test_wilcoxon_returns_nan_for_n_lt_5() -> None:
    a = np.array([1.0, 2.0])
    b = np.array([0.0, 1.0])
    stat, p = paired_wilcoxon_one_sided_greater(a, b)
    assert np.isnan(stat) and np.isnan(p)
