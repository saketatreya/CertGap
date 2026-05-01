"""Soft Actor-Critic training with policy-improvement diagnostic logging.

The SAC path is intentionally scoped to continuous-control Gymnasium tasks.
It logs checkpoint-level return changes and critic Bellman residuals, which
lets the empirical audit test whether off-policy TD error tracks policy
improvement. We do not report an off-policy surrogate advantage; `A_k` and
`cert_gap_k` are logged as NaN until a principled estimator is added.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import optim

from certgap.common.networks import PolicyNet, ValueNet
from certgap.common.utils import (
    get_device,
    infer_env_spec,
    make_env,
    runtime_metadata,
    set_global_seeds,
)


@dataclass
class SACConfig:
    total_timesteps: int = 200_000
    gamma: float = 0.99
    tau: float = 0.005
    alpha: float = 0.2
    batch_size: int = 256
    replay_size: int = 500_000
    learning_starts: int = 5_000
    train_freq: int = 4
    policy_lr: float = 3e-4
    q_lr: float = 3e-4
    hidden_sizes: tuple[int, int] = (64, 64)
    checkpoint_interval: int = 5_000
    eval_episodes: int = 3
    residual_batch_size: int = 2048
    start_state_action_samples: int = 4
    log_every_checkpoints: int = 1

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["hidden_sizes"] = list(self.hidden_sizes)
        return out


class ReplayBuffer:
    def __init__(self, state_dim: int, action_dim: int, capacity: int, device: torch.device) -> None:
        self.capacity = int(capacity)
        self.device = device
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.pos = 0
        self.size = 0

    def add(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        self.states[self.pos] = np.asarray(state, dtype=np.float32).reshape(-1)
        self.actions[self.pos] = np.asarray(action, dtype=np.float32).reshape(-1)
        self.rewards[self.pos] = float(reward)
        self.next_states[self.pos] = np.asarray(next_state, dtype=np.float32).reshape(-1)
        self.dones[self.pos] = float(done)
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> tuple[torch.Tensor, ...]:
        if self.size == 0:
            raise ValueError("Cannot sample from an empty replay buffer.")
        idx = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.as_tensor(self.states[idx], dtype=torch.float32, device=self.device),
            torch.as_tensor(self.actions[idx], dtype=torch.float32, device=self.device),
            torch.as_tensor(self.rewards[idx], dtype=torch.float32, device=self.device),
            torch.as_tensor(self.next_states[idx], dtype=torch.float32, device=self.device),
            torch.as_tensor(self.dones[idx], dtype=torch.float32, device=self.device),
        )


def _build_policy_net(env_spec, device: torch.device, hidden: tuple[int, int]) -> PolicyNet:
    action_low = torch.as_tensor(env_spec.action_low, dtype=torch.float32, device=device)
    action_high = torch.as_tensor(env_spec.action_high, dtype=torch.float32, device=device)
    return PolicyNet(
        env_spec.state_dim,
        env_spec.action_dim,
        action_kind="box",
        action_low=action_low,
        action_high=action_high,
        hidden=hidden,
    ).to(device)


def _q_input(states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
    return torch.cat([states, actions], dim=-1)


def _sample_policy(
    policy: PolicyNet, states: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    dist = policy.distribution(states)
    actions = dist.rsample()
    log_probs = dist.log_prob(actions)
    return actions, log_probs


def _deterministic_action(policy: PolicyNet, state: np.ndarray, device: torch.device) -> np.ndarray:
    state_t = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        mean = policy.dist_info(state_t)["mean"]
        action = torch.tanh(mean) * policy.action_scale + policy.action_bias
    return action.squeeze(0).cpu().numpy().astype(np.float32)


def _stochastic_action(policy: PolicyNet, state: np.ndarray, device: torch.device) -> np.ndarray:
    state_t = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        action = policy.distribution(state_t).sample()
    return action.squeeze(0).cpu().numpy().astype(np.float32)


def _soft_update(target: torch.nn.Module, source: torch.nn.Module, tau: float) -> None:
    with torch.no_grad():
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.mul_(1.0 - tau).add_(source_param, alpha=tau)


def _update_sac(
    policy: PolicyNet,
    q1: ValueNet,
    q2: ValueNet,
    target_q1: ValueNet,
    target_q2: ValueNet,
    policy_optim: optim.Optimizer,
    q_optim: optim.Optimizer,
    replay: ReplayBuffer,
    cfg: SACConfig,
) -> dict[str, float]:
    states, actions, rewards, next_states, dones = replay.sample(cfg.batch_size)

    with torch.no_grad():
        next_actions, next_log_probs = _sample_policy(policy, next_states)
        target_q = torch.minimum(
            target_q1(_q_input(next_states, next_actions)).squeeze(-1),
            target_q2(_q_input(next_states, next_actions)).squeeze(-1),
        )
        target = rewards + cfg.gamma * (1.0 - dones) * (target_q - cfg.alpha * next_log_probs)

    q1_pred = q1(_q_input(states, actions)).squeeze(-1)
    q2_pred = q2(_q_input(states, actions)).squeeze(-1)
    q_loss = F.mse_loss(q1_pred, target) + F.mse_loss(q2_pred, target)
    q_optim.zero_grad()
    q_loss.backward()
    q_optim.step()

    new_actions, log_probs = _sample_policy(policy, states)
    q_new = torch.minimum(
        q1(_q_input(states, new_actions)).squeeze(-1),
        q2(_q_input(states, new_actions)).squeeze(-1),
    )
    policy_loss = (cfg.alpha * log_probs - q_new).mean()
    policy_optim.zero_grad()
    policy_loss.backward()
    policy_optim.step()

    _soft_update(target_q1, q1, cfg.tau)
    _soft_update(target_q2, q2, cfg.tau)

    entropy = float((-log_probs).mean().item())
    return {
        "q_loss": float(q_loss.item()),
        "policy_loss": float(policy_loss.item()),
        "policy_entropy": entropy,
    }


def _evaluate_policy(
    env_id: str,
    policy: PolicyNet,
    device: torch.device,
    seed: int,
    n_episodes: int,
) -> tuple[float, np.ndarray]:
    env, _, _ = make_env(env_id, seed)
    returns: list[float] = []
    start_states: list[np.ndarray] = []
    for episode in range(n_episodes):
        obs, _ = env.reset(seed=seed + episode)
        start_states.append(np.asarray(obs, dtype=np.float32).reshape(-1))
        done = False
        total = 0.0
        while not done:
            action = _deterministic_action(policy, obs, device)
            obs, reward, terminated, truncated, _ = env.step(action)
            total += float(reward)
            done = bool(terminated or truncated)
        returns.append(total)
    env.close()
    return float(np.mean(returns)), np.stack(start_states)


def _estimate_start_state_value(
    policy: PolicyNet,
    q1: ValueNet,
    q2: ValueNet,
    start_states: np.ndarray,
    cfg: SACConfig,
    device: torch.device,
) -> float:
    states = torch.as_tensor(start_states, dtype=torch.float32, device=device)
    estimates: list[torch.Tensor] = []
    with torch.no_grad():
        for _ in range(cfg.start_state_action_samples):
            actions, log_probs = _sample_policy(policy, states)
            q_val = torch.minimum(
                q1(_q_input(states, actions)).squeeze(-1),
                q2(_q_input(states, actions)).squeeze(-1),
            )
            estimates.append(q_val - cfg.alpha * log_probs)
        stacked = torch.stack(estimates, dim=0)
    return float(stacked.mean().item())


def _compute_residuals(
    policy: PolicyNet,
    q1: ValueNet,
    q2: ValueNet,
    target_q1: ValueNet,
    target_q2: ValueNet,
    replay: ReplayBuffer,
    cfg: SACConfig,
) -> dict[str, float]:
    batch_size = min(cfg.residual_batch_size, replay.size)
    states, actions, rewards, next_states, dones = replay.sample(batch_size)
    with torch.no_grad():
        next_actions, next_log_probs = _sample_policy(policy, next_states)
        target_q = torch.minimum(
            target_q1(_q_input(next_states, next_actions)).squeeze(-1),
            target_q2(_q_input(next_states, next_actions)).squeeze(-1),
        )
        target = rewards + cfg.gamma * (1.0 - dones) * (target_q - cfg.alpha * next_log_probs)
        res1 = target - q1(_q_input(states, actions)).squeeze(-1)
        res2 = target - q2(_q_input(states, actions)).squeeze(-1)
        residuals = torch.cat([res1, res2], dim=0)
        abs_res = residuals.abs()
        return {
            "eps_u_mse": float(residuals.pow(2).mean().item()),
            "eps_u_sup": float(abs_res.max().item()),
            "eps_u_p95": float(torch.quantile(abs_res, 0.95).item()),
        }


def _empty_log() -> dict[str, list[float]]:
    keys = [
        "update_idx",
        "timesteps",
        "J_k",
        "delta_J_k",
        "harmful_k",
        "eps_u",
        "eps_u_mse",
        "eps_u_p95",
        "eps_u_sup",
        "delta_hat_k",
        "A_k",
        "cert_gap_k",
        "q_loss",
        "policy_loss",
        "policy_entropy",
        "alpha",
        "replay_size",
        "n_eval_episodes",
    ]
    return {key: [] for key in keys}


def train_sac(
    env_id: str,
    seed: int,
    cfg: SACConfig | None = None,
    *,
    verbose: bool = True,
) -> dict[str, Any]:
    """Run one SAC seed and return checkpoint-level diagnostic logs."""
    cfg = cfg or SACConfig()
    set_global_seeds(seed)
    device = get_device()
    env, resolved_env_id, warnings = make_env(env_id, seed)
    env_spec = infer_env_spec(env)
    if env_spec.action_kind != "box":
        env.close()
        raise ValueError("SAC implementation supports continuous Box action spaces only.")

    policy = _build_policy_net(env_spec, device, cfg.hidden_sizes)
    q1 = ValueNet(env_spec.state_dim + env_spec.action_dim, hidden=cfg.hidden_sizes).to(device)
    q2 = ValueNet(env_spec.state_dim + env_spec.action_dim, hidden=cfg.hidden_sizes).to(device)
    target_q1 = ValueNet(env_spec.state_dim + env_spec.action_dim, hidden=cfg.hidden_sizes).to(device)
    target_q2 = ValueNet(env_spec.state_dim + env_spec.action_dim, hidden=cfg.hidden_sizes).to(device)
    target_q1.load_state_dict(q1.state_dict())
    target_q2.load_state_dict(q2.state_dict())

    policy_optim = optim.Adam(policy.parameters(), lr=cfg.policy_lr)
    q_optim = optim.Adam(list(q1.parameters()) + list(q2.parameters()), lr=cfg.q_lr)
    replay = ReplayBuffer(env_spec.state_dim, env_spec.action_dim, cfg.replay_size, device)

    obs, _ = env.reset(seed=seed)
    log = _empty_log()
    last_losses = {"q_loss": float("nan"), "policy_loss": float("nan"), "policy_entropy": float("nan")}
    prev_J: float | None = None
    start_time = time.time()
    checkpoint_idx = 0

    for step in range(1, cfg.total_timesteps + 1):
        if step <= cfg.learning_starts:
            action = env.action_space.sample()
        else:
            action = _stochastic_action(policy, obs, device)

        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = bool(terminated or truncated)
        replay.add(obs, action, float(reward), next_obs, done)
        obs = next_obs
        if done:
            obs, _ = env.reset()

        if replay.size >= cfg.batch_size and step > cfg.learning_starts and step % cfg.train_freq == 0:
            last_losses = _update_sac(
                policy,
                q1,
                q2,
                target_q1,
                target_q2,
                policy_optim,
                q_optim,
                replay,
                cfg,
            )

        if step % cfg.checkpoint_interval == 0 or step == cfg.total_timesteps:
            eval_seed = seed + 10_000 + checkpoint_idx * 100
            J_k, start_states = _evaluate_policy(env_id, policy, device, eval_seed, cfg.eval_episodes)
            residuals = (
                _compute_residuals(policy, q1, q2, target_q1, target_q2, replay, cfg)
                if replay.size > 0
                else {"eps_u_mse": float("nan"), "eps_u_sup": float("nan"), "eps_u_p95": float("nan")}
            )
            start_value = _estimate_start_state_value(policy, q1, q2, start_states, cfg, device)
            delta_hat_k = J_k - start_value

            log["update_idx"].append(checkpoint_idx)
            log["timesteps"].append(step)
            log["J_k"].append(float(J_k))
            log["delta_J_k"].append(float("nan"))
            log["harmful_k"].append(float("nan"))
            log["eps_u"].append(float(residuals["eps_u_mse"]))
            log["eps_u_mse"].append(float(residuals["eps_u_mse"]))
            log["eps_u_p95"].append(float(residuals["eps_u_p95"]))
            log["eps_u_sup"].append(float(residuals["eps_u_sup"]))
            log["delta_hat_k"].append(float(delta_hat_k))
            log["A_k"].append(float("nan"))
            log["cert_gap_k"].append(float("nan"))
            log["q_loss"].append(float(last_losses["q_loss"]))
            log["policy_loss"].append(float(last_losses["policy_loss"]))
            log["policy_entropy"].append(float(last_losses["policy_entropy"]))
            log["alpha"].append(float(cfg.alpha))
            log["replay_size"].append(float(replay.size))
            log["n_eval_episodes"].append(float(cfg.eval_episodes))

            if prev_J is not None and len(log["J_k"]) > 1:
                delta = J_k - prev_J
                log["delta_J_k"][-2] = float(delta)
                log["harmful_k"][-2] = float(delta < 0)
            prev_J = J_k

            if verbose and (
                checkpoint_idx % cfg.log_every_checkpoints == 0
                or step == cfg.total_timesteps
            ):
                print(
                    f"[SAC {env_id} seed {seed} | step {step:6d}] "
                    f"J={J_k:8.2f} eps_mse={residuals['eps_u_mse']:.4f} "
                    f"delta_hat={delta_hat_k:8.2f} replay={replay.size}"
                )
            checkpoint_idx += 1

    env.close()
    runtime_seconds = time.time() - start_time
    metadata = runtime_metadata(
        env_id=env_id,
        resolved_env_id=resolved_env_id,
        seed=seed,
        config=cfg.to_dict(),
        runtime_seconds=runtime_seconds,
        spec=env_spec,
        warnings=warnings,
    )
    metadata.update(
        {
            "algo": "sac",
            "n_updates": len(log["update_idx"]),
            "total_timesteps": cfg.total_timesteps,
            "log_keys": list(log.keys()),
        }
    )

    log_np = {
        key: np.asarray(values, dtype=np.int64 if key in {"update_idx", "timesteps"} else float)
        for key, values in log.items()
    }
    return {"metadata": metadata, "log": log_np}


__all__ = ["ReplayBuffer", "SACConfig", "train_sac"]
