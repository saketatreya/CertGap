"""TRPO-style on-policy training with the same diagnostic log schema as PPO.

This implementation is intentionally compact and used for the optional pilot
grid. It uses a sampled KL trust region, conjugate gradients, and backtracking
line search. Continuous-action KL is computed in the pre-tanh Gaussian space,
which is the standard practical approximation for squashed Gaussian policies.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import optim

from certgap.common.metrics import (
    compute_eps_u,
    compute_surrogate_advantage,
    per_update_log_keys,
)
from certgap.common.networks import PolicyNet, ValueNet
from certgap.common.rollout import RolloutBatch, collect_rollout
from certgap.common.utils import (
    EnvSpec,
    get_device,
    infer_env_spec,
    iter_minibatches,
    make_env,
    runtime_metadata,
    set_global_seeds,
)


@dataclass
class TRPOConfig:
    gamma: float = 0.99
    gae_lambda: float = 0.95
    total_timesteps: int = 150_000
    n_steps: int = 2048
    minibatch_size: int = 64
    value_epochs: int = 10
    value_lr: float = 1e-3
    target_kl: float = 0.01
    damping: float = 0.1
    cg_iters: int = 10
    line_search_steps: int = 10
    max_grad_norm: float = 1.0
    eps_u_kind: str = "mse"
    hidden_sizes: tuple[int, int] = (64, 64)
    log_every_updates: int = 10

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["hidden_sizes"] = list(self.hidden_sizes)
        return out


def _build_policy_net(env_spec: EnvSpec, device: torch.device, hidden: tuple[int, int]) -> PolicyNet:
    action_low = (
        torch.as_tensor(env_spec.action_low, dtype=torch.float32, device=device)
        if env_spec.action_low is not None
        else None
    )
    action_high = (
        torch.as_tensor(env_spec.action_high, dtype=torch.float32, device=device)
        if env_spec.action_high is not None
        else None
    )
    return PolicyNet(
        env_spec.state_dim,
        env_spec.action_dim,
        action_kind=env_spec.action_kind,
        action_low=action_low,
        action_high=action_high,
        hidden=hidden,
    ).to(device)


def _flat_params(model: torch.nn.Module) -> torch.Tensor:
    return torch.cat([p.data.view(-1) for p in model.parameters()])


def _set_flat_params(model: torch.nn.Module, flat: torch.Tensor) -> None:
    idx = 0
    for param in model.parameters():
        n = param.numel()
        param.data.copy_(flat[idx : idx + n].view_as(param))
        idx += n


def _flat_grad(grads: tuple[torch.Tensor | None, ...]) -> torch.Tensor:
    chunks = []
    for grad in grads:
        if grad is not None:
            chunks.append(grad.contiguous().view(-1))
    return torch.cat(chunks)


def _surrogate(policy: PolicyNet, batch: RolloutBatch, advantages: torch.Tensor) -> torch.Tensor:
    new_log_probs = policy.log_prob(batch.states, batch.actions)
    ratio = torch.exp(new_log_probs - batch.log_probs.detach())
    return (ratio * advantages).mean()


def _mean_kl(policy: PolicyNet, batch: RolloutBatch) -> torch.Tensor:
    if batch.action_kind == "discrete":
        old_probs = batch.probs.detach().clamp_min(1e-8)  # type: ignore[union-attr]
        new_probs = policy.probs(batch.states).clamp_min(1e-8)
        return (old_probs * (old_probs.log() - new_probs.log())).sum(dim=-1).mean()

    old_mean = batch.dist_info["mean"].detach()
    old_log_std = batch.dist_info["log_std"].detach()
    new_info = policy.dist_info(batch.states)
    new_mean = new_info["mean"]
    new_log_std = new_info["log_std"]
    old_var = torch.exp(2.0 * old_log_std)
    new_var = torch.exp(2.0 * new_log_std)
    kl = new_log_std - old_log_std + (old_var + (old_mean - new_mean).pow(2)) / (2.0 * new_var) - 0.5
    return kl.sum(dim=-1).mean()


def _fisher_vector_product(
    policy: PolicyNet,
    batch: RolloutBatch,
    vector: torch.Tensor,
    damping: float,
) -> torch.Tensor:
    kl = _mean_kl(policy, batch)
    grads = torch.autograd.grad(kl, policy.parameters(), create_graph=True)
    flat_grad_kl = _flat_grad(grads)
    grad_vector = torch.dot(flat_grad_kl, vector)
    hvp = torch.autograd.grad(grad_vector, policy.parameters())
    return _flat_grad(hvp).detach() + damping * vector


def _conjugate_gradient(fn, b: torch.Tensor, iters: int, tol: float = 1e-10) -> torch.Tensor:
    x = torch.zeros_like(b)
    r = b.clone()
    p = b.clone()
    r_dot = torch.dot(r, r)
    for _ in range(iters):
        z = fn(p)
        alpha = r_dot / (torch.dot(p, z) + 1e-12)
        x += alpha * p
        r -= alpha * z
        new_r_dot = torch.dot(r, r)
        if new_r_dot < tol:
            break
        beta = new_r_dot / (r_dot + 1e-12)
        p = r + beta * p
        r_dot = new_r_dot
    return x


def _trpo_policy_step(policy: PolicyNet, batch: RolloutBatch, cfg: TRPOConfig) -> float:
    advantages = batch.advantages.detach()
    if torch.std(advantages) > 1e-8:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    objective = _surrogate(policy, batch, advantages)
    grads = torch.autograd.grad(objective, policy.parameters())
    grad = _flat_grad(grads).detach()
    if not torch.isfinite(grad).all() or torch.norm(grad) < 1e-12:
        return float("nan")

    fvp = lambda v: _fisher_vector_product(policy, batch, v, cfg.damping)
    step_dir = _conjugate_gradient(fvp, grad, cfg.cg_iters)
    shs = 0.5 * torch.dot(step_dir, fvp(step_dir))
    if not torch.isfinite(shs) or shs <= 0:
        return float("nan")
    step = step_dir * torch.sqrt(torch.as_tensor(cfg.target_kl, device=grad.device) / (shs + 1e-12))

    old_params = _flat_params(policy).clone()
    old_objective = float(objective.detach().item())
    accepted_kl = float("nan")
    for frac in [0.5**i for i in range(cfg.line_search_steps)]:
        _set_flat_params(policy, old_params + frac * step)
        with torch.no_grad():
            new_objective = float(_surrogate(policy, batch, advantages).item())
            kl = float(_mean_kl(policy, batch).item())
        if np.isfinite(new_objective) and np.isfinite(kl) and kl <= cfg.target_kl and new_objective >= old_objective:
            accepted_kl = kl
            break
    else:
        _set_flat_params(policy, old_params)
    return accepted_kl


def _value_step(
    value_net: ValueNet,
    optimizer: optim.Optimizer,
    states: torch.Tensor,
    returns: torch.Tensor,
    epochs: int,
    minibatch_size: int,
) -> float:
    last_loss = 0.0
    for _ in range(epochs):
        epoch_loss = 0.0
        n = 0
        for batch_indices in iter_minibatches(states.shape[0], minibatch_size):
            idx = torch.as_tensor(batch_indices, dtype=torch.long, device=states.device)
            preds = value_net(states[idx]).squeeze(-1)
            loss = F.mse_loss(preds, returns[idx])
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(value_net.parameters(), 1.0)
            optimizer.step()
            epoch_loss += float(loss.item())
            n += 1
        if n > 0:
            last_loss = epoch_loss / n
    return last_loss


def _value_loss_on_batch(value_net: ValueNet, states: torch.Tensor, returns: torch.Tensor) -> float:
    with torch.no_grad():
        preds = value_net(states).squeeze(-1)
        return float(F.mse_loss(preds, returns).item())


def _start_state_indices(dones: torch.Tensor) -> torch.Tensor:
    if dones.shape[0] == 0:
        return torch.zeros(0, dtype=torch.long, device=dones.device)
    after_done = (dones[:-1] > 0.5).nonzero(as_tuple=True)[0] + 1
    starts = torch.cat([torch.zeros(1, dtype=torch.long, device=dones.device), after_done])
    return starts[starts < dones.shape[0]]


def _delta_hat(value_net: ValueNet, states: torch.Tensor, dones: torch.Tensor, mean_return: float) -> float:
    start_idx = _start_state_indices(dones)
    if start_idx.numel() == 0:
        return float("nan")
    with torch.no_grad():
        v_starts = value_net(states[start_idx]).squeeze(-1).mean().item()
    return float(mean_return) - float(v_starts)


def _mean_episode_return(batch: RolloutBatch) -> float:
    if batch.episode_returns:
        return float(np.mean(batch.episode_returns))
    return float(batch.current_episode_return)


def _empty_log() -> dict[str, list[float]]:
    keys = per_update_log_keys()
    keys += ["approx_kl"]
    return {key: [] for key in keys}


def train_trpo(
    env_id: str,
    seed: int,
    cfg: TRPOConfig | None = None,
    *,
    verbose: bool = True,
) -> dict[str, Any]:
    cfg = cfg or TRPOConfig()
    set_global_seeds(seed)
    device = get_device()
    env, resolved_env_id, warnings = make_env(env_id, seed)
    env_spec = infer_env_spec(env)

    policy_net = _build_policy_net(env_spec, device, cfg.hidden_sizes)
    value_net = ValueNet(env_spec.state_dim, hidden=cfg.hidden_sizes).to(device)
    value_optim = optim.Adam(value_net.parameters(), lr=cfg.value_lr)

    observation, _ = env.reset(seed=seed)
    current_episode_return = 0.0
    current_episode_length = 0
    log = _empty_log()
    total_updates = int(np.ceil(cfg.total_timesteps / cfg.n_steps))
    total_steps = 0
    prev_J: float | None = None
    start_time = time.time()

    for update in range(total_updates):
        batch, observation, current_episode_return, current_episode_length = collect_rollout(
            env=env,
            policy_net=policy_net,
            value_net=value_net,
            n_steps=cfg.n_steps,
            gamma=cfg.gamma,
            gae_lambda=cfg.gae_lambda,
            device=device,
            observation=observation,
            current_episode_return=current_episode_return,
            current_episode_length=current_episode_length,
        )
        total_steps += int(batch.states.shape[0])
        J_k = _mean_episode_return(batch)

        approx_kl = _trpo_policy_step(policy_net, batch, cfg)
        with torch.no_grad():
            new_log_probs = policy_net.log_prob(batch.states, batch.actions)
            A_k = compute_surrogate_advantage(batch.log_probs.detach(), new_log_probs, batch.advantages)
            policy_entropy = float(policy_net.entropy(batch.states).mean().item())

        value_loss_pre = _value_loss_on_batch(value_net, batch.states, batch.returns)
        _value_step(value_net, value_optim, batch.states, batch.returns, cfg.value_epochs, cfg.minibatch_size)
        value_loss_post = _value_loss_on_batch(value_net, batch.states, batch.returns)
        eps_u = compute_eps_u(
            value_net,
            batch.states,
            batch.next_states,
            batch.rewards,
            batch.dones,
            cfg.gamma,
            kind=cfg.eps_u_kind,
        )
        delta_hat_k = _delta_hat(value_net, batch.states, batch.dones, J_k)
        cert_gap_k = A_k - delta_hat_k

        log["update_idx"].append(update)
        log["timesteps"].append(total_steps)
        log["A_k"].append(float(A_k))
        log["delta_hat_k"].append(float(delta_hat_k))
        log["cert_gap_k"].append(float(cert_gap_k))
        log["eps_u"].append(float(eps_u))
        log["J_k"].append(float(J_k))
        log["delta_J_k"].append(float("nan"))
        log["harmful_k"].append(float("nan"))
        log["clip_frac"].append(float("nan"))
        log["policy_entropy"].append(float(policy_entropy))
        log["value_loss_pre"].append(float(value_loss_pre))
        log["value_loss_post"].append(float(value_loss_post))
        log["n_episodes"].append(int(len(batch.episode_returns)))
        log["approx_kl"].append(float(approx_kl))

        if prev_J is not None and update > 0:
            delta = J_k - prev_J
            log["delta_J_k"][-2] = float(delta)
            log["harmful_k"][-2] = float(delta < 0)
        prev_J = J_k

        if verbose and ((update + 1) % cfg.log_every_updates == 0 or update == 0 or update == total_updates - 1):
            print(
                f"[TRPO {env_id} seed {seed} | step {total_steps:6d}] "
                f"J={J_k:8.2f} A_k={A_k:7.3f} delta_hat={delta_hat_k:7.3f} "
                f"eps_u={eps_u:7.3f} kl={approx_kl:7.4f}"
            )

    runtime_seconds = time.time() - start_time
    env.close()
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
            "algo": "trpo",
            "n_updates": total_updates,
            "total_timesteps": total_steps,
            "log_keys": list(log.keys()),
        }
    )
    log_np = {
        key: np.asarray(values, dtype=np.int64 if key in {"n_episodes", "update_idx"} else float)
        for key, values in log.items()
    }
    return {"metadata": metadata, "log": log_np}


__all__ = ["TRPOConfig", "train_trpo"]
