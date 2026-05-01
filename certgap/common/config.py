"""PPO configuration aligned to the paper.

Defaults match `paper.pdf` Appendix H ("PPO configuration") exactly:
γ = 0.99, GAE λ = 0.95, value-epochs = 10, policy-epochs = 10,
clip-ε = 0.2, batch = 2048, minibatch = 64, policy-lr = 3e-4,
value-lr = 1e-3, grad-clip = 1.0, no entropy bonus.

Continuous-action environments use a Gaussian policy with state-independent
log-std (paper Appendix H).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class PPOConfig:
    # Discounting / GAE
    gamma: float = 0.99
    gae_lambda: float = 0.95

    # Optimization budget
    total_timesteps: int = 500_000
    n_steps: int = 2048
    minibatch_size: int = 64
    policy_epochs: int = 10
    value_epochs: int = 10

    # PPO knobs
    clip_eps: float = 0.2
    entropy_coef: float = 0.0  # paper: "no entropy bonus"
    value_coef: float = 0.5
    policy_lr: float = 3e-4
    value_lr: float = 1e-3
    max_grad_norm: float = 1.0

    # Bellman-residual scalar used for the single `eps_u` log field.
    # The dedicated PPO-MSE grid records the canonical MSE residual.
    eps_u_kind: str = "p95"  # one of {"p95", "mse", "sup"}

    # Held-out batch size for residual-estimator agreement checks; 0 disables.
    heldout_batch_size: int = 0

    # Architecture / device
    hidden_sizes: tuple[int, int] = (64, 64)
    device: str = "cpu"

    # Intervention
    gating_threshold: float | None = None

    # Logging cadence (per-update log is always written; this controls stdout)
    log_every_updates: int = 10

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["hidden_sizes"] = list(self.hidden_sizes)
        return d


@dataclass
class RunConfig:
    """Identifies a single (env, seed, output) triple."""

    env_id: str
    seed: int
    output_path: str
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
