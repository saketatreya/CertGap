"""Per-update metrics: ε_u, A_k, Δ̂_k, policy-change scalars.

The headline ε_u definition in the paper (Appendix H) is **mean squared
TD error on the rollout post-critic-update**. Two alternative definitions
are reported in figB to head off "you used a robust estimator" objections:

- "mse"  — mean of squared Bellman residuals  (paper headline; default)
- "sup"  — sup-norm of |Bellman residual|     (matches the theoretical bound)
- "p95"  — 95th percentile of |Bellman residual|  (robust variant)

Every PPO update computes all three when `log_eps_u_variants=True`; the
single-scalar `eps_u` reported in main figures uses `eps_u_kind` from
the config.
"""

from __future__ import annotations

from typing import Any

import torch

EPS_U_KINDS = ("mse", "sup", "p95")


def compute_value_residuals(
    value_net: torch.nn.Module,
    states: torch.Tensor,
    next_states: torch.Tensor,
    rewards: torch.Tensor,
    dones: torch.Tensor,
    gamma: float,
) -> torch.Tensor:
    """One-step TD residual: r + γ(1-d)V(s') − V(s)."""
    with torch.no_grad():
        values = value_net(states).squeeze(-1)
        next_values = value_net(next_states).squeeze(-1)
        targets = rewards + gamma * (1.0 - dones) * next_values
        residuals = targets - values
    return residuals


def compute_eps_u(
    value_net: torch.nn.Module,
    states: torch.Tensor,
    next_states: torch.Tensor,
    rewards: torch.Tensor,
    dones: torch.Tensor,
    gamma: float,
    *,
    kind: str = "mse",
    percentile: float = 0.95,
) -> float:
    """Bellman-residual scalar under one of {mse, sup, p95}.

    The paper's analytic bound uses the sup-norm; the paper's logged
    headline scalar uses MSE. Both reduce to the same qualitative result
    (AUROC ≈ 0.5 across all 8 envs) per figB.
    """
    if kind not in EPS_U_KINDS:
        raise ValueError(f"eps_u kind must be one of {EPS_U_KINDS}, got {kind!r}")
    residuals = compute_value_residuals(value_net, states, next_states, rewards, dones, gamma)
    if kind == "mse":
        return float(residuals.pow(2).mean().item())
    if kind == "sup":
        return float(residuals.abs().max().item())
    return float(torch.quantile(residuals.abs(), percentile).item())


def compute_eps_u_all(
    value_net: torch.nn.Module,
    states: torch.Tensor,
    next_states: torch.Tensor,
    rewards: torch.Tensor,
    dones: torch.Tensor,
    gamma: float,
) -> dict[str, float]:
    """Return all three ε_u variants from a single forward pass."""
    residuals = compute_value_residuals(value_net, states, next_states, rewards, dones, gamma)
    abs_res = residuals.abs()
    return {
        "mse": float(residuals.pow(2).mean().item()),
        "sup": float(abs_res.max().item()),
        "p95": float(torch.quantile(abs_res, 0.95).item()),
    }


def compute_id_advancement(
    value_net: torch.nn.Module,
    states: torch.Tensor,
    next_states: torch.Tensor,
    rewards: torch.Tensor,
    dones: torch.Tensor,
    gamma: float,
) -> float:
    """Δ̂ contribution from one batch: mean Bellman residual / (1 − γ).

    This is the per-batch building block of Δ̂_k. The full Δ̂_k = Ĵ(π_k) −
    (1/|S_0|) Σ V_{k+1}(s_0) is computed in `train_ppo` directly because
    it needs Ĵ(π_k) (mean episode return) and V_{k+1} on initial states.
    """
    with torch.no_grad():
        v_s = value_net(states).squeeze(-1)
        v_next = value_net(next_states).squeeze(-1)
        residuals = rewards + gamma * (1.0 - dones) * v_next - v_s
        mean_res = float(residuals.mean().item())
    return mean_res / (1.0 - gamma)


def compute_surrogate_advantage(
    old_log_probs: torch.Tensor,
    new_log_probs: torch.Tensor,
    advantages: torch.Tensor,
) -> float:
    """A_k via importance-weighted GAE on the on-policy rollout.

    A_k := E_{(s,a)~B_k}[ (π_{k+1}(a|s) / π_k(a|s)) Â_GAE(s, a) ]
        ≈ mean_i [ exp(log_π_new − log_π_old) · A_GAE_i ]   on the batch.
    """
    with torch.no_grad():
        ratio = (new_log_probs - old_log_probs).exp()
        return float((ratio * advantages).mean().item())


def compute_policy_change_tv(old_probs: torch.Tensor, new_probs: torch.Tensor) -> float:
    """C_k := E_s ||π_{k+1}(·|s) − π_k(·|s)||_1.  Discrete only."""
    return float((old_probs - new_probs).abs().sum(dim=-1).mean().item())


def compute_policy_change_kl(
    old_probs: torch.Tensor, new_probs: torch.Tensor, eps: float = 1e-8
) -> float:
    old = old_probs.clamp_min(eps)
    new = new_probs.clamp_min(eps)
    return float((old * (old.log() - new.log())).sum(dim=-1).mean().item())


def compute_policy_entropy_discrete(probs: torch.Tensor, eps: float = 1e-8) -> float:
    probs = probs.clamp_min(eps)
    return float(-(probs * probs.log()).sum(dim=-1).mean().item())


def per_update_log_keys(
    *, log_heldout: bool = False, log_eps_u_variants: bool = False
) -> list[str]:
    """Canonical schema. Every key here ends up as a same-length 1-D array
    in the per-seed pickle. Length = number of PPO updates in the run.
    """
    keys: list[str] = [
        "update_idx",       # 0, 1, 2, …
        "timesteps",        # cumulative env steps at update boundary
        "A_k",              # surrogate advantage
        "delta_hat_k",      # critic bias at start states  Ĵ(π_k) − E_ν V_{k+1}(s_0)
        "cert_gap_k",       # A_k − delta_hat_k
        "eps_u",            # headline (kind from PPOConfig)
        "J_k",              # mean episode return on rollout k
        "delta_J_k",        # J_{k+1} − J_k  (NaN on the last update)
        "harmful_k",        # int(delta_J_k < 0)  (NaN on the last update)
        "clip_frac",
        "policy_entropy",
        "value_loss_pre",
        "value_loss_post",
        "grad_norm",
        "explained_var",
        "rejected",
        "n_episodes",
    ]
    if log_eps_u_variants:
        keys += ["eps_u_mse", "eps_u_sup", "eps_u_p95"]
    if log_heldout:
        keys += ["delta_hat_train_k", "delta_hat_heldout_k", "J_heldout_k"]
    return keys


__all__ = [
    "EPS_U_KINDS",
    "compute_eps_u",
    "compute_eps_u_all",
    "compute_id_advancement",
    "compute_policy_change_kl",
    "compute_policy_change_tv",
    "compute_policy_entropy_discrete",
    "compute_surrogate_advantage",
    "compute_value_residuals",
    "per_update_log_keys",
]
