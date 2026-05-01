"""On-policy rollout collection with GAE.

The held-out batch in `train_ppo(log_heldout=True)` is just a second
call to `collect_rollout` with `n_steps=heldout_batch_size` against a
separately-seeded eval env, using the same policy and value networks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from certgap.common.gae import compute_gae


@dataclass
class RolloutBatch:
    states: torch.Tensor
    next_states: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor
    values: torch.Tensor
    next_values: torch.Tensor
    log_probs: torch.Tensor
    probs: torch.Tensor | None
    dist_info: dict[str, torch.Tensor]
    action_kind: str
    advantages: torch.Tensor
    returns: torch.Tensor
    episode_returns: list[float]
    current_episode_return: float
    current_episode_length: int


def collect_rollout(
    env: Any,
    policy_net: torch.nn.Module,
    value_net: torch.nn.Module,
    n_steps: int,
    gamma: float,
    gae_lambda: float,
    device: torch.device,
    observation: np.ndarray | None = None,
    current_episode_return: float = 0.0,
    current_episode_length: int = 0,
) -> tuple[RolloutBatch, np.ndarray, float, int]:
    if observation is None:
        observation, _ = env.reset()

    states: list[np.ndarray] = []
    next_states: list[np.ndarray] = []
    actions: list[Any] = []
    rewards: list[float] = []
    dones: list[float] = []
    values: list[float] = []
    log_probs: list[float] = []
    probs: list[np.ndarray] = []
    means: list[np.ndarray] = []
    log_stds: list[np.ndarray] = []
    episode_returns: list[float] = []

    action_kind = getattr(policy_net, "action_kind", "discrete")

    for _ in range(n_steps):
        state_tensor = torch.as_tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            distribution = policy_net.distribution(state_tensor)
            action_tensor = distribution.sample()
            if action_kind == "discrete":
                action = int(action_tensor.item())
            else:
                action = action_tensor.squeeze(0).cpu().numpy().astype(np.float32)
            log_prob = float(policy_net.log_prob(state_tensor, action_tensor).item())
            value = float(value_net(state_tensor).squeeze(-1).item())
            dist_info = policy_net.dist_info(state_tensor)

        next_observation, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        states.append(np.asarray(observation, dtype=np.float32))
        next_states.append(np.asarray(next_observation, dtype=np.float32))
        actions.append(action)
        rewards.append(float(reward))
        dones.append(float(done))
        values.append(value)
        log_probs.append(log_prob)
        if action_kind == "discrete":
            probs.append(dist_info["probs"].squeeze(0).cpu().numpy().astype(np.float32))
        else:
            means.append(dist_info["mean"].squeeze(0).detach().cpu().numpy().astype(np.float32))
            log_stds.append(dist_info["log_std"].squeeze(0).detach().cpu().numpy().astype(np.float32))

        current_episode_return += float(reward)
        current_episode_length += 1

        if done:
            episode_returns.append(current_episode_return)
            observation, _ = env.reset()
            current_episode_return = 0.0
            current_episode_length = 0
        else:
            observation = next_observation

    states_tensor = torch.as_tensor(np.stack(states), dtype=torch.float32, device=device)
    next_states_tensor = torch.as_tensor(np.stack(next_states), dtype=torch.float32, device=device)
    rewards_tensor = torch.as_tensor(rewards, dtype=torch.float32, device=device)
    dones_tensor = torch.as_tensor(dones, dtype=torch.float32, device=device)
    if action_kind == "discrete":
        actions_tensor = torch.as_tensor(actions, dtype=torch.long, device=device)
        probs_tensor = torch.as_tensor(np.stack(probs), dtype=torch.float32, device=device)
        dist_info = {"probs": probs_tensor}
    else:
        actions_tensor = torch.as_tensor(np.stack(actions), dtype=torch.float32, device=device)
        probs_tensor = None
        dist_info = {
            "mean": torch.as_tensor(np.stack(means), dtype=torch.float32, device=device),
            "log_std": torch.as_tensor(np.stack(log_stds), dtype=torch.float32, device=device),
        }
    values_tensor = torch.as_tensor(values, dtype=torch.float32, device=device)
    log_probs_tensor = torch.as_tensor(log_probs, dtype=torch.float32, device=device)

    with torch.no_grad():
        next_values_tensor = value_net(next_states_tensor).squeeze(-1)
    advantages, returns = compute_gae(
        rewards=rewards_tensor,
        dones=dones_tensor,
        values=values_tensor,
        next_values=next_values_tensor,
        gamma=gamma,
        gae_lambda=gae_lambda,
    )

    batch = RolloutBatch(
        states=states_tensor,
        next_states=next_states_tensor,
        actions=actions_tensor,
        rewards=rewards_tensor,
        dones=dones_tensor,
        values=values_tensor,
        next_values=next_values_tensor,
        log_probs=log_probs_tensor,
        probs=probs_tensor,
        dist_info=dist_info,
        action_kind=action_kind,
        advantages=advantages,
        returns=returns,
        episode_returns=episode_returns,
        current_episode_return=current_episode_return,
        current_episode_length=current_episode_length,
    )
    return batch, observation, current_episode_return, current_episode_length
