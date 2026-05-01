"""Shared loaders for residual-vs-improvement audit tables and figures."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np

from certgap.analysis.harm_prediction import auroc


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


AUDIT_SOURCES: list[dict[str, Any]] = [
    {"algo": "PPO", "roots": [Path("results/ppo"), Path("results/main")]},
    {"algo": "PPO-MSE", "roots": [Path("results/ppo_mse")]},
    {"algo": "SAC", "roots": [Path("results/sac")]},
    {"algo": "TRPO", "roots": [Path("results/trpo")]},
]


def _first_existing_root(roots: list[Path]) -> Path | None:
    for root in roots:
        if root.exists() and list(root.glob("*/*.pkl")):
            return root
    return None


def _seed_paths(root: Path, env: str) -> list[Path]:
    env_dir = root / env
    if not env_dir.exists():
        return []
    return sorted(env_dir.glob("seed_*.pkl"))


def _load(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        return pickle.load(handle)


def _per_seed_auc(paths: list[Path], key: str, sign: int) -> tuple[float, int]:
    vals = []
    for path in paths:
        log = _load(path)["log"]
        if key not in log:
            continue
        metric = sign * np.asarray(log[key], dtype=float)
        harm = np.asarray(log["harmful_k"], dtype=float)
        val = auroc(metric, harm)
        if np.isfinite(val):
            vals.append(val)
    if not vals:
        return float("nan"), 0
    return float(np.median(vals)), len(vals)


def collect_audit_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source in AUDIT_SOURCES:
        root = _first_existing_root(source["roots"])
        if root is None:
            continue
        envs = ENV_ORDER if source["algo"] != "SAC" else ["Hopper-v5", "HalfCheetah-v5", "Walker2d-v5"]
        if source["algo"] == "TRPO":
            envs = ["LunarLander-v3", "Hopper-v5"]
        for env in envs:
            paths = _seed_paths(root, env)
            if not paths:
                continue
            eps, eps_n = _per_seed_auc(paths, "eps_u", -1)
            eps_mse, eps_mse_n = _per_seed_auc(paths, "eps_u_mse", -1)
            eps_p95, eps_p95_n = _per_seed_auc(paths, "eps_u_p95", -1)
            delta_hat, delta_hat_n = _per_seed_auc(paths, "delta_hat_k", -1)
            rows.append(
                {
                    "algorithm": source["algo"],
                    "env": env,
                    "root": str(root),
                    "n_runs": len(paths),
                    "n_valid_eps_u": eps_n,
                    "auroc_neg_eps_u": eps,
                    "n_valid_eps_u_mse": eps_mse_n,
                    "auroc_neg_eps_u_mse": eps_mse,
                    "n_valid_eps_u_p95": eps_p95_n,
                    "auroc_neg_eps_u_p95": eps_p95,
                    "n_valid_neg_delta_hat": delta_hat_n,
                    "auroc_neg_delta_hat": delta_hat,
                }
            )
    return rows


__all__ = ["AUDIT_SOURCES", "ENV_ORDER", "collect_audit_rows"]
