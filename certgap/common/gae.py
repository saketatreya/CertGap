"""Generalized Advantage Estimation (Schulman et al., 2015b)."""

from __future__ import annotations

import torch


def compute_gae(
    rewards: torch.Tensor,
    dones: torch.Tensor,
    values: torch.Tensor,
    next_values: torch.Tensor,
    gamma: float,
    gae_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    advantages = torch.zeros_like(rewards)
    gae = torch.zeros((), dtype=rewards.dtype, device=rewards.device)
    for step in reversed(range(rewards.shape[0])):
        mask = 1.0 - dones[step]
        delta = rewards[step] + gamma * next_values[step] * mask - values[step]
        gae = delta + gamma * gae_lambda * mask * gae
        advantages[step] = gae
    returns = advantages + values
    return advantages, returns
