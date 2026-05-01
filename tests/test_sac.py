"""Small SAC utility tests."""

from __future__ import annotations

import numpy as np
import torch

from certgap.sac import ReplayBuffer


def test_replay_buffer_sample_shapes() -> None:
    buffer = ReplayBuffer(state_dim=3, action_dim=2, capacity=16, device=torch.device("cpu"))
    for i in range(10):
        buffer.add(
            state=np.array([i, i + 1, i + 2], dtype=np.float32),
            action=np.array([0.1, -0.1], dtype=np.float32),
            reward=float(i),
            next_state=np.array([i + 1, i + 2, i + 3], dtype=np.float32),
            done=i % 2 == 0,
        )

    states, actions, rewards, next_states, dones = buffer.sample(4)
    assert states.shape == (4, 3)
    assert actions.shape == (4, 2)
    assert rewards.shape == (4,)
    assert next_states.shape == (4, 3)
    assert dones.shape == (4,)
