"""Single PPO training loop with paper-aligned per-update logging.

Replaces slop's `train_ppo` + `train_ppo_diagnostic` + `train_ppo_factorial`
(1123 lines, three near-duplicate functions, scattered controller logic)
with one ~400-line function. All logged quantities follow paper.pdf
Section 2 / Appendix H definitions exactly.

Per-update log schema (canonical) — see `metrics.per_update_log_keys`.

Key definitions (paper §2):

    Â_k     = E_{(s,a)~B_k}[ (π_{k+1}(a|s) / π_k(a|s)) · Â^GAE_{V_k}(s, a) ]
    Δ̂_k    = Ĵ(π_k)  −  (1/|S_0|) Σ_{s_0 ∈ S_0} V_{k+1}(s_0)
    cĝ_k   = Â_k − Δ̂_k
    ε̂_u   = mean squared TD error on rollout transitions, post-critic-update
    ΔJ_k   = Ĵ(π_{k+1}) − Ĵ(π_k)        (lagged: NaN on the final update)

`Δ̂_k` here uses the start-state form (Ĵ(π_k) − E[V_{k+1}(s_0)]), which
differs from the residual-based estimator `mean(r + γ V(s') − V(s)) / (1−γ)`.
The held-out comparison logs the residual-based estimator on the on-policy
rollout and on a separately seeded held-out batch.

`log_heldout=True` activates that path; the per-update log adds
`delta_hat_train_k` and `delta_hat_heldout_k` (both residual / (1−γ)).
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import optim

from certgap.common.config import PPOConfig
from certgap.common.metrics import (
    compute_eps_u,
    compute_eps_u_all,
    compute_policy_change_tv,
    compute_surrogate_advantage,
    compute_value_residuals,
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


# --------------------------------------------------------------------- helpers


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


def _ppo_policy_step(
    policy_net: PolicyNet,
    optimizer: optim.Optimizer,
    batch: RolloutBatch,
    cfg: PPOConfig,
) -> float:
    """Run `policy_epochs` of PPO clipped surrogate updates. Returns the
    fraction of samples whose ratio left the [1−clip_eps, 1+clip_eps] band
    on the *first* epoch (paper-standard `clip_frac` proxy)."""
    batch_size = batch.actions.shape[0]
    advantages = batch.advantages.detach()
    old_log_probs = batch.log_probs.detach()
    first_epoch_clip_frac: float | None = None

    for epoch in range(cfg.policy_epochs):
        for batch_indices in iter_minibatches(batch_size, cfg.minibatch_size):
            idx = torch.as_tensor(batch_indices, dtype=torch.long, device=batch.states.device)
            new_log_probs = policy_net.log_prob(batch.states[idx], batch.actions[idx])
            ratio = (new_log_probs - old_log_probs[idx]).exp()
            unclipped = ratio * advantages[idx]
            clipped = torch.clamp(ratio, 1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps) * advantages[idx]
            policy_loss = -torch.minimum(unclipped, clipped).mean()
            if cfg.entropy_coef > 0.0:
                entropy = policy_net.entropy(batch.states[idx]).mean()
                loss = policy_loss - cfg.entropy_coef * entropy
            else:
                loss = policy_loss
            optimizer.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(policy_net.parameters(), cfg.max_grad_norm)
            optimizer.step()
            if epoch == 0 and first_epoch_clip_frac is None:
                with torch.no_grad():
                    first_epoch_clip_frac = float(
                        (torch.abs(ratio.detach() - 1.0) > cfg.clip_eps).float().mean().item()
                    )
    return float(first_epoch_clip_frac or 0.0), float(grad_norm.item())


def _value_step(
    value_net: ValueNet,
    optimizer: optim.Optimizer,
    states: torch.Tensor,
    returns: torch.Tensor,
    epochs: int,
    minibatch_size: int,
) -> float:
    """MSE regression of V(s) on GAE returns. Returns final-epoch mean loss."""
    last_loss = 0.0
    for epoch in range(epochs):
        epoch_loss = 0.0
        n = 0
        for batch_indices in iter_minibatches(states.shape[0], minibatch_size):
            idx = torch.as_tensor(batch_indices, dtype=torch.long, device=states.device)
            preds = value_net(states[idx]).squeeze(-1)
            loss = F.mse_loss(preds, returns[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item())
            n += 1
        if n > 0:
            last_loss = epoch_loss / n
    return last_loss

def _explained_variance(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Computes fraction of variance that y_pred explains about y_true.
    Returns 1 - var(y_true - y_pred) / var(y_true).
    """
    var_y = np.var(y_true)
    if var_y < 1e-8:
        return 0.0
    return 1 - np.var(y_true - y_pred) / var_y


def _value_loss_on_batch(value_net: ValueNet, states: torch.Tensor, returns: torch.Tensor) -> float:
    with torch.no_grad():
        preds = value_net(states).squeeze(-1)
        return float(F.mse_loss(preds, returns).item())


def _start_state_indices(dones: torch.Tensor) -> torch.Tensor:
    """Indices into a rollout batch where a new episode began.

    Returns the index right after each `done==1` plus index 0. These are
    `S_0` in the paper's Δ̂_k definition (states sampled from ν).
    """
    n = dones.shape[0]
    if n == 0:
        return torch.zeros(0, dtype=torch.long, device=dones.device)
    after_done = (dones[:-1] > 0.5).nonzero(as_tuple=True)[0] + 1
    starts = torch.cat([torch.zeros(1, dtype=torch.long, device=dones.device), after_done])
    return starts[starts < n]


def _delta_hat_paper(
    value_net: ValueNet,
    states: torch.Tensor,
    dones: torch.Tensor,
    mean_episode_return: float,
) -> float:
    """Paper definition: Δ̂_k = Ĵ(π_k) − E_{ν}[V_{k+1}(s_0)]."""
    start_idx = _start_state_indices(dones)
    if start_idx.numel() == 0:
        return float("nan")
    with torch.no_grad():
        v_starts = value_net(states[start_idx]).squeeze(-1).mean().item()
    return float(mean_episode_return) - float(v_starts)


def _delta_hat_residual(
    value_net: ValueNet, batch: RolloutBatch, gamma: float
) -> float:
    """Residual-based estimator: mean(r + γV(s') − V(s)) / (1 − γ).

    This is the estimator at risk of "overfitting on the rollout that
    just trained the critic." figA compares it on B_k_train vs B_k_heldout.
    """
    residuals = compute_value_residuals(
        value_net, batch.states, batch.next_states, batch.rewards, batch.dones, gamma
    )
    return float(residuals.mean().item()) / (1.0 - gamma)


def _mean_episode_return(batch: RolloutBatch) -> float:
    if batch.episode_returns:
        return float(np.mean(batch.episode_returns))
    return float(batch.current_episode_return)


def _empty_log(keys: list[str]) -> dict[str, list[float]]:
    return {k: [] for k in keys}


# --------------------------------------------------------------------- main


def train_ppo(
    env_id: str,
    seed: int,
    cfg: PPOConfig | None = None,
    *,
    log_heldout: bool = False,
    log_eps_u_variants: bool = False,
    factorial_value_epochs: int | None = None,
    factorial_clip_eps: float | None = None,
    factorial_value_lr: float | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """Run one PPO training seed and return per-update logs + metadata.

    Per-update log keys are exactly `metrics.per_update_log_keys(...)`.
    Episode-level data is reported via `J_k`, `delta_J_k`, `harmful_k`,
    `n_episodes` (number of complete episodes that ended in rollout k).

    `log_heldout` adds `delta_hat_train_k`, `delta_hat_heldout_k`, and `J_heldout_k` —
    both residual-based estimators (paper Δ̂ uses start states; the
    held-out comparison is about the residual estimator's overfitting
    risk, which is the actual reviewer concern).

    `log_eps_u_variants` adds `eps_u_mse`, `eps_u_sup`, `eps_u_p95`.

    Factorial overrides take effect after defaults are merged from `cfg`.
    """
    cfg = cfg or PPOConfig()
    if factorial_value_epochs is not None:
        cfg.value_epochs = factorial_value_epochs
    if factorial_clip_eps is not None:
        cfg.clip_eps = factorial_clip_eps
    if factorial_value_lr is not None:
        cfg.value_lr = factorial_value_lr

    set_global_seeds(seed)
    device = get_device()
    env, resolved_env_id, warnings = make_env(env_id, seed)
    env_spec = infer_env_spec(env)

    # Held-out env uses a disjoint seed and a disjoint random stream.
    heldout_env = None
    heldout_obs = None
    heldout_steps = cfg.heldout_batch_size if log_heldout else 0
    if log_heldout:
        if heldout_steps <= 0:
            heldout_steps = 500
        heldout_env, _, _ = make_env(env_id, seed + 1_000_000)
        heldout_obs, _ = heldout_env.reset(seed=seed + 1_000_000)

    policy_net = _build_policy_net(env_spec, device, cfg.hidden_sizes)
    value_net = ValueNet(env_spec.state_dim, hidden=cfg.hidden_sizes).to(device)
    policy_optim = optim.Adam(policy_net.parameters(), lr=cfg.policy_lr)
    value_optim = optim.Adam(value_net.parameters(), lr=cfg.value_lr)

    observation, _ = env.reset(seed=seed)
    current_episode_return = 0.0
    current_episode_length = 0
    heldout_eps_return = 0.0
    heldout_eps_length = 0

    keys = per_update_log_keys(
        log_heldout=log_heldout, log_eps_u_variants=log_eps_u_variants
    )
    log = _empty_log(keys)

    total_updates = int(np.ceil(cfg.total_timesteps / cfg.n_steps))
    total_steps = 0
    start_time = time.time()

    prev_J: float | None = None
    last_J: float | None = None  # used to fill the lagged ΔJ for the final-but-one update

    heartbeat_path = Path("results/.heartbeats") / f"{env_id}_seed{seed}.json"
    heartbeat_path.parent.mkdir(parents=True, exist_ok=True)

    for update in range(total_updates):
        # Update heartbeat
        if (update % 5 == 0) or update == total_updates - 1:
            try:
                with heartbeat_path.open("w") as f:
                    json.dump({
                        "env_id": env_id,
                        "seed": seed,
                        "update": update,
                        "total_updates": total_updates,
                        "steps": total_steps,
                        "total_timesteps": cfg.total_timesteps,
                        "J_k": float(last_J) if last_J is not None else 0.0,
                        "time": time.time(),
                    }, f)
            except Exception:
                pass

        # 1. Collect rollout B_k under π_k with critic V_k.
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
        n_episodes = len(batch.episode_returns)

        # 2. PPO policy update → π_{k+1}.
        if cfg.gating_threshold is not None:
            policy_state_before = {k: v.clone() for k, v in policy_net.state_dict().items()}
            value_state_before = {k: v.clone() for k, v in value_net.state_dict().items()}

        clip_frac, grad_norm = _ppo_policy_step(policy_net, policy_optim, batch, cfg)

        # 3. A_k and policy entropy under π_{k+1}.
        with torch.no_grad():
            new_log_probs = policy_net.log_prob(batch.states, batch.actions)
            A_k = compute_surrogate_advantage(batch.log_probs.detach(), new_log_probs, batch.advantages)
            policy_entropy = float(policy_net.entropy(batch.states).mean().item())

        # 4. Value update → V_{k+1}.
        value_loss_pre = _value_loss_on_batch(value_net, batch.states, batch.returns)
        _ = _value_step(
            value_net,
            value_optim,
            batch.states,
            batch.returns,
            cfg.value_epochs,
            cfg.minibatch_size,
        )
        value_loss_post = _value_loss_on_batch(value_net, batch.states, batch.returns)

        # 5. ε_u under V_{k+1} on the rollout. Paper: post-critic-update.
        with torch.no_grad():
            v_post = value_net(batch.states).squeeze(-1).cpu().numpy()
            explained_var = _explained_variance(v_post, batch.returns.cpu().numpy())
        eps_u_headline = compute_eps_u(
            value_net,
            batch.states,
            batch.next_states,
            batch.rewards,
            batch.dones,
            cfg.gamma,
            kind=cfg.eps_u_kind,
        )
        eps_u_dict: dict[str, float] | None = None
        if log_eps_u_variants:
            eps_u_dict = compute_eps_u_all(
                value_net,
                batch.states,
                batch.next_states,
                batch.rewards,
                batch.dones,
                cfg.gamma,
            )

        # 6. Δ̂_k (paper definition) and cert-gap.
        delta_hat_k = _delta_hat_paper(value_net, batch.states, batch.dones, J_k)
        
        # Gating intervention
        rejected = False
        if cfg.gating_threshold is not None and delta_hat_k > cfg.gating_threshold:
            policy_net.load_state_dict(policy_state_before)
            value_net.load_state_dict(value_state_before)
            rejected = True
            # Re-evaluate delta_hat after rollback
            delta_hat_k = _delta_hat_paper(value_net, batch.states, batch.dones, J_k)

        cert_gap_k = A_k - delta_hat_k

        # 7. Held-out residual-based Δ̂ for figA.
        delta_hat_train_k: float | None = None
        delta_hat_heldout_k: float | None = None
        if log_heldout:
            delta_hat_train_k = _delta_hat_residual(value_net, batch, cfg.gamma)
            heldout_batch, heldout_obs, heldout_eps_return, heldout_eps_length = collect_rollout(
                env=heldout_env,
                policy_net=policy_net,
                value_net=value_net,
                n_steps=heldout_steps,
                gamma=cfg.gamma,
                gae_lambda=cfg.gae_lambda,
                device=device,
                observation=heldout_obs,
                current_episode_return=heldout_eps_return,
                current_episode_length=heldout_eps_length,
            )
            delta_hat_heldout_k = _delta_hat_residual(value_net, heldout_batch, cfg.gamma)
            J_heldout_k = _mean_episode_return(heldout_batch)

        # 8. Lag-one ΔJ: filled in at *next* update; here we tag prev row.
        log["update_idx"].append(update)
        log["timesteps"].append(total_steps)
        log["A_k"].append(float(A_k))
        log["delta_hat_k"].append(float(delta_hat_k))
        log["cert_gap_k"].append(float(cert_gap_k))
        log["eps_u"].append(float(eps_u_headline))
        log["J_k"].append(float(J_k))
        log["delta_J_k"].append(float("nan"))   # patched in next step
        log["harmful_k"].append(float("nan"))   # patched in next step
        log["clip_frac"].append(float(clip_frac))
        log["policy_entropy"].append(float(policy_entropy))
        log["value_loss_pre"].append(float(value_loss_pre))
        log["value_loss_post"].append(float(value_loss_post))
        log["grad_norm"].append(float(grad_norm))
        log["explained_var"].append(float(explained_var))
        log["rejected"].append(float(rejected))
        log["n_episodes"].append(int(n_episodes))
        if log_eps_u_variants and eps_u_dict is not None:
            log["eps_u_mse"].append(float(eps_u_dict["mse"]))
            log["eps_u_sup"].append(float(eps_u_dict["sup"]))
            log["eps_u_p95"].append(float(eps_u_dict["p95"]))
        if log_heldout:
            log["delta_hat_train_k"].append(float(delta_hat_train_k))      # type: ignore[arg-type]
            log["delta_hat_heldout_k"].append(float(delta_hat_heldout_k))  # type: ignore[arg-type]
            log["J_heldout_k"].append(float(J_heldout_k))                 # type: ignore[arg-type]

        # patch prev row's lag-one ΔJ
        if prev_J is not None and update > 0:
            delta = J_k - prev_J
            log["delta_J_k"][-2] = float(delta)
            log["harmful_k"][-2] = float(delta < 0)
        prev_J = J_k
        last_J = J_k

        if verbose and (
            (update + 1) % cfg.log_every_updates == 0
            or update == 0
            or update == total_updates - 1
        ):
            print(
                f"[{env_id} seed {seed} | step {total_steps:6d}] "
                f"J={J_k:8.2f}  A_k={A_k:7.3f}  Δ̂_k={delta_hat_k:7.3f}  "
                f"cg={cert_gap_k:7.3f}  ε_u={eps_u_headline:7.3f}  "
                f"clip_frac={clip_frac:.2f}  H={policy_entropy:.2f}"
            )

    # The final update has no successor rollout. Leave last `delta_J_k`
    # and `harmful_k` as NaN — every analysis script must filter these.
    runtime_seconds = time.time() - start_time
    env.close()
    if heldout_env is not None:
        heldout_env.close()
    
    if heartbeat_path.exists():
        heartbeat_path.unlink()

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
            "algo": "ppo",
            "log_heldout": log_heldout,
            "log_eps_u_variants": log_eps_u_variants,
            "factorial_overrides": {
                "value_epochs": factorial_value_epochs,
                "clip_eps": factorial_clip_eps,
                "value_lr": factorial_value_lr,
            },
            "n_updates": total_updates,
            "total_timesteps": total_steps,
            "log_keys": keys,
        }
    )

    # Convert log lists to numpy arrays for downstream analysis.
    log_np = {k: np.asarray(v, dtype=float if k != "n_episodes" and k != "update_idx" else np.int64) for k, v in log.items()}
    return {"metadata": metadata, "log": log_np}


__all__ = ["train_ppo"]
