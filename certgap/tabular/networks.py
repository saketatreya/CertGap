"""Small neural networks for the Bridge experiment.

Actor:  [S → 32 → 32 → A] softmax, one-hot state input.
Critic: [S → H → H → A]   linear output (Q-network).
  - Underparameterized: H=16  (cannot represent 256 arbitrary Q-values)
  - Overparameterized:  H=128 (can represent them easily)
"""
from __future__ import annotations

import math

import torch
from torch import nn


class BridgeActor(nn.Module):
    """Small softmax policy for tabular MDP with one-hot states."""

    def __init__(self, n_states: int, n_actions: int, hidden: int = 32) -> None:
        super().__init__()
        self.n_states = n_states
        self.n_actions = n_actions
        self.net = nn.Sequential(
            nn.Linear(n_states, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, n_actions),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2.0))
                nn.init.zeros_(m.bias)
        # Small init for last layer
        last = [m for m in self.net if isinstance(m, nn.Linear)][-1]
        nn.init.orthogonal_(last.weight, gain=0.01)

    def forward(self, one_hot: torch.Tensor) -> torch.Tensor:
        """Returns logits, shape (..., A)."""
        return self.net(one_hot)

    def probs(self, one_hot: torch.Tensor) -> torch.Tensor:
        """Returns action probabilities, shape (..., A)."""
        return torch.softmax(self.forward(one_hot), dim=-1)

    def get_policy_table(self) -> torch.Tensor:
        """Returns full policy table pi(a|s), shape (S, A)."""
        with torch.no_grad():
            eye = torch.eye(self.n_states, device=next(self.parameters()).device)
            return self.probs(eye)


class BridgeCritic(nn.Module):
    """V-network for tabular MDP with one-hot states."""

    def __init__(self, n_states: int, hidden: int = 16) -> None:
        super().__init__()
        self.n_states = n_states
        self.net = nn.Sequential(
            nn.Linear(n_states, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2.0))
                nn.init.zeros_(m.bias)
        last = [m for m in self.net if isinstance(m, nn.Linear)][-1]
        nn.init.orthogonal_(last.weight, gain=0.01)

    def forward(self, one_hot: torch.Tensor) -> torch.Tensor:
        """Returns V-values, shape (..., 1)."""
        return self.net(one_hot)

    def get_V_table(self) -> torch.Tensor:
        """Returns full V-table f(s), shape (S,)."""
        with torch.no_grad():
            eye = torch.eye(self.n_states, device=next(self.parameters()).device)
            return self.forward(eye).squeeze(-1)
