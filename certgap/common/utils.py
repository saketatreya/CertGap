"""Shared utilities: env factory, seeding, env spec inference, save/load."""

from __future__ import annotations

import json
import os
import pickle
import random
import subprocess
import sys
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces


# --------------------------------------------------------------------- env spec


class EnvSpec:
    __slots__ = ("state_dim", "action_dim", "action_kind", "action_low", "action_high")

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        action_kind: str,
        action_low: np.ndarray | None = None,
        action_high: np.ndarray | None = None,
    ) -> None:
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_kind = action_kind
        self.action_low = action_low
        self.action_high = action_high


def infer_env_spec(env: gym.Env) -> EnvSpec:
    if not hasattr(env.observation_space, "shape") or env.observation_space.shape is None:
        raise ValueError("Observation space must expose a finite shape.")
    state_dim = int(np.prod(env.observation_space.shape))
    action_space = env.action_space
    if isinstance(action_space, spaces.Discrete):
        return EnvSpec(state_dim=state_dim, action_dim=int(action_space.n), action_kind="discrete")
    if isinstance(action_space, spaces.Box):
        if action_space.shape is None:
            raise ValueError("Box action space must expose a finite shape.")
        low = np.asarray(action_space.low, dtype=np.float32).reshape(-1)
        high = np.asarray(action_space.high, dtype=np.float32).reshape(-1)
        if (not np.all(np.isfinite(low)) or not np.all(np.isfinite(high))) and hasattr(env, "unwrapped"):
            base_space = getattr(env.unwrapped, "action_space", None)
            if isinstance(base_space, spaces.Box):
                low = np.asarray(base_space.low, dtype=np.float32).reshape(-1)
                high = np.asarray(base_space.high, dtype=np.float32).reshape(-1)
        if not np.all(np.isfinite(low)) or not np.all(np.isfinite(high)):
            raise ValueError("Box action bounds must be finite for continuous PPO support.")
        return EnvSpec(
            state_dim=state_dim,
            action_dim=int(np.prod(action_space.shape)),
            action_kind="box",
            action_low=low,
            action_high=high,
        )
    raise ValueError(f"Unsupported action space type: {type(env.action_space).__name__}")


# --------------------------------------------------------------------- env make


_ENV_ALIASES: dict[str, list[str]] = {
    "LunarLander-v3": ["LunarLander-v3", "LunarLander-v2"],
    "LunarLander-v2": ["LunarLander-v2", "LunarLander-v3"],
    "Hopper-v5": ["Hopper-v5", "Hopper-v4"],
    "Walker2d-v5": ["Walker2d-v5", "Walker2d-v4"],
    "HalfCheetah-v5": ["HalfCheetah-v5", "HalfCheetah-v4"],
    "Ant-v5": ["Ant-v5", "Ant-v4"],
    "Humanoid-v5": ["Humanoid-v5", "Humanoid-v4"],
}


def make_env(env_id: str, seed: int) -> tuple[gym.Env, str, list[str]]:
    """Create env, falling back across version aliases. Box action spaces
    are wrapped with `ClipAction` to keep continuous-action PPO well-behaved.
    Returns (env, resolved_id, warnings).
    """
    warnings: list[str] = []
    last_error: Exception | None = None
    for candidate in _ENV_ALIASES.get(env_id, [env_id]):
        try:
            env = gym.make(candidate)
            if isinstance(env.action_space, spaces.Box):
                env = gym.wrappers.ClipAction(env)
            env.reset(seed=seed)
            if candidate != env_id:
                warnings.append(f"Requested {env_id!r} resolved to {candidate!r}.")
            return env, candidate, warnings
        except Exception as exc:
            warnings.append(f"Failed to create env {candidate!r}: {exc}")
            last_error = exc
    raise RuntimeError(
        f"Unable to create env {env_id!r}. Errors: {' | '.join(warnings)}"
    ) from last_error


# --------------------------------------------------------------------- seeding


def get_device() -> torch.device:
    return torch.device("cpu")


def set_global_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# --------------------------------------------------------------------- io


def ensure_dir(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def save_pickle(data: Any, path: str | Path) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("wb") as handle:
        pickle.dump(data, handle)
    return path


def load_pickle(path: str | Path) -> Any:
    with Path(path).open("rb") as handle:
        return pickle.load(handle)


def save_json(data: Any, path: str | Path) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True, default=_json_default)
    return path


def _json_default(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Not JSON serializable: {type(value).__name__}")


# --------------------------------------------------------------------- runtime


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def current_git_head(cwd: str | Path | None = None) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(cwd) if cwd is not None else None,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def runtime_metadata(
    *,
    env_id: str,
    resolved_env_id: str,
    seed: int,
    config: dict[str, Any],
    runtime_seconds: float,
    spec: EnvSpec,
    warnings: list[str],
) -> dict[str, Any]:
    return {
        "timestamp_utc": utc_timestamp(),
        "git_head": current_git_head(),
        "env_id": env_id,
        "resolved_env_id": resolved_env_id,
        "seed": seed,
        "runtime_seconds": runtime_seconds,
        "state_dim": spec.state_dim,
        "action_dim": spec.action_dim,
        "action_kind": spec.action_kind,
        "warnings": warnings,
        "config": config,
        "python": sys.version.split()[0],
        "torch": getattr(torch, "__version__", "unknown"),
        "gymnasium": getattr(gym, "__version__", "unknown"),
        "cwd": os.getcwd(),
    }


# --------------------------------------------------------------------- minibatch


def iter_minibatches(num_items: int, batch_size: int, shuffle: bool = True) -> Iterator[np.ndarray]:
    indices = np.arange(num_items)
    if shuffle:
        np.random.shuffle(indices)
    for start in range(0, num_items, batch_size):
        yield indices[start : start + batch_size]
