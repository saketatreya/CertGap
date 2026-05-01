"""Train vs held-out Δ̂ agreement check.

Reads `delta_hat_train_k` and `delta_hat_heldout_k` from a per-seed log
(produced by `train_ppo(log_heldout=True)`) and fits an OLS regression
of the held-out estimator on the train estimator. Reports Pearson r,
OLS slope, intercept, and R².

The two quantities are different empirical estimators of the same
population Δ_k. The slope tells us whether they agree in expectation
(slope ≈ 1 ⇒ unbiased relative to each other); R² tells us the
per-update agreement (high ⇒ the residual estimator is not overfit
to the rollout that trained the critic).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from certgap.analysis.correlations import pearson


@dataclass
class AgreementResult:
    n_pairs: int
    pearson_r: float
    pearson_r_lo: float
    pearson_r_hi: float
    slope: float
    intercept: float
    r_squared: float
    train: np.ndarray
    heldout: np.ndarray


def fit_train_heldout_regression(train: np.ndarray, heldout: np.ndarray) -> AgreementResult:
    train = np.asarray(train, dtype=float).ravel()
    heldout = np.asarray(heldout, dtype=float).ravel()
    mask = np.isfinite(train) & np.isfinite(heldout)
    train = train[mask]
    heldout = heldout[mask]
    if train.size < 10:
        return AgreementResult(
            n_pairs=int(train.size),
            pearson_r=float("nan"),
            pearson_r_lo=float("nan"),
            pearson_r_hi=float("nan"),
            slope=float("nan"),
            intercept=float("nan"),
            r_squared=float("nan"),
            train=train,
            heldout=heldout,
        )
    pr = pearson(train, heldout)
    x_mean = train.mean()
    y_mean = heldout.mean()
    cov = np.mean((train - x_mean) * (heldout - y_mean))
    var = np.mean((train - x_mean) ** 2)
    slope = float(cov / var) if var > 0 else float("nan")
    intercept = float(y_mean - slope * x_mean) if np.isfinite(slope) else float("nan")
    if np.isfinite(slope):
        residuals = heldout - (intercept + slope * train)
        ss_res = float(np.sum(residuals ** 2))
        ss_tot = float(np.sum((heldout - y_mean) ** 2))
        r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    else:
        r_squared = float("nan")
    return AgreementResult(
        n_pairs=int(train.size),
        pearson_r=pr.r,
        pearson_r_lo=pr.lo,
        pearson_r_hi=pr.hi,
        slope=slope,
        intercept=intercept,
        r_squared=float(r_squared),
        train=train,
        heldout=heldout,
    )


def collect_pairs(pkl_paths: list[str | Path]) -> tuple[np.ndarray, np.ndarray]:
    """Concatenate `delta_hat_train_k` / `delta_hat_heldout_k` across seeds."""
    import pickle

    train_chunks: list[np.ndarray] = []
    heldout_chunks: list[np.ndarray] = []
    for pkl in pkl_paths:
        with open(pkl, "rb") as f:
            data = pickle.load(f)
        log = data["log"]
        if "delta_hat_train_k" not in log or "delta_hat_heldout_k" not in log:
            raise ValueError(
                f"{pkl} was not produced with --log-heldout; rerun with that flag."
            )
        train_chunks.append(np.asarray(log["delta_hat_train_k"], dtype=float))
        heldout_chunks.append(np.asarray(log["delta_hat_heldout_k"], dtype=float))
    return np.concatenate(train_chunks), np.concatenate(heldout_chunks)
