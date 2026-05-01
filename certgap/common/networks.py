"""Policy and value networks. (64, 64)-Tanh MLPs, paper Appendix H.

Continuous-action policies use a Gaussian with state-independent log-std,
squashed via tanh and affinely mapped onto the env's action box (paper
Appendix H: "continuous-action environments use a Gaussian policy with
state-independent log-standard-deviation").
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.distributions import (
    AffineTransform,
    Categorical,
    Independent,
    Normal,
    TanhTransform,
    TransformedDistribution,
)


def _init_linear(layer: nn.Linear, gain: float) -> None:
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.zeros_(layer.bias)


def _mlp(in_dim: int, hidden: tuple[int, int]) -> nn.Sequential:
    h1, h2 = hidden
    return nn.Sequential(
        nn.Linear(in_dim, h1),
        nn.Tanh(),
        nn.Linear(h1, h2),
        nn.Tanh(),
    )


class PolicyNet(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        *,
        action_kind: str = "discrete",
        action_low: torch.Tensor | None = None,
        action_high: torch.Tensor | None = None,
        log_std_init: float = -0.5,
        hidden: tuple[int, int] = (64, 64),
    ) -> None:
        super().__init__()
        self.action_kind = action_kind
        if action_kind == "discrete":
            self.backbone = _mlp(state_dim, hidden)
            self.head = nn.Linear(hidden[-1], action_dim)
        elif action_kind == "box":
            self.backbone = _mlp(state_dim, hidden)
            self.head = nn.Linear(hidden[-1], action_dim)  # mean head
            self.log_std = nn.Parameter(torch.full((action_dim,), float(log_std_init)))
            if action_low is None or action_high is None:
                raise ValueError("Continuous-action PolicyNet requires action_low and action_high.")
            self.register_buffer("action_low", action_low.clone().detach().float())
            self.register_buffer("action_high", action_high.clone().detach().float())
            self.register_buffer("action_scale", 0.5 * (self.action_high - self.action_low))
            self.register_buffer("action_bias", 0.5 * (self.action_high + self.action_low))
        else:
            raise ValueError(f"Unsupported action_kind: {action_kind!r}")
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for layer in self.backbone:
            if isinstance(layer, nn.Linear):
                _init_linear(layer, math.sqrt(2.0))
        _init_linear(self.head, 0.01)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(states))

    def probs(self, states: torch.Tensor) -> torch.Tensor:
        if self.action_kind != "discrete":
            raise ValueError("PolicyNet.probs is only defined for discrete actions.")
        return torch.softmax(self(states), dim=-1)

    def dist_info(self, states: torch.Tensor) -> dict[str, torch.Tensor]:
        if self.action_kind == "discrete":
            logits = self(states)
            return {"logits": logits, "probs": torch.softmax(logits, dim=-1)}
        mean = self(states)
        log_std = self.log_std.unsqueeze(0).expand_as(mean)
        return {"mean": mean, "log_std": log_std}

    def distribution(self, states: torch.Tensor):
        if self.action_kind == "discrete":
            return Categorical(logits=self(states))
        mean = self(states)
        log_std = self.log_std.unsqueeze(0).expand_as(mean)
        base = Independent(Normal(mean, log_std.exp()), 1)
        transforms: list = [TanhTransform(cache_size=1)]
        if not torch.allclose(self.action_scale, torch.ones_like(self.action_scale)) or not torch.allclose(
            self.action_bias, torch.zeros_like(self.action_bias)
        ):
            transforms.append(AffineTransform(loc=self.action_bias, scale=self.action_scale))
        return TransformedDistribution(base, transforms)

    def log_prob(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        return self.distribution(states).log_prob(actions)

    def entropy(self, states: torch.Tensor) -> torch.Tensor:
        if self.action_kind == "discrete":
            return self.distribution(states).entropy()
        # Closed-form Gaussian entropy on the pre-tanh latent. Off by a constant
        # under tanh squashing; only used for diagnostics.
        log_std = self.log_std
        return (log_std + 0.5 * (1.0 + math.log(2.0 * math.pi))).sum().expand(states.shape[0])


class ValueNet(nn.Module):
    def __init__(self, state_dim: int, *, hidden: tuple[int, int] = (64, 64)) -> None:
        super().__init__()
        self.model = nn.Sequential(
            *_mlp(state_dim, hidden),
            nn.Linear(hidden[-1], 1),
        )
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        linears = [layer for layer in self.model if isinstance(layer, nn.Linear)]
        for layer in linears[:-1]:
            _init_linear(layer, math.sqrt(2.0))
        _init_linear(linears[-1], 0.01)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        return self.model(states)
