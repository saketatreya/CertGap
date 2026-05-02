"""End-to-end PPO smoke: tiny CartPole run, validate the log schema."""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest

from certgap.common.config import PPOConfig
from certgap.common.metrics import per_update_log_keys


@pytest.mark.slow
def test_tiny_cartpole_run() -> None:
    """~30 seconds; guarded with `-m slow` so default `pytest` skips it."""
    pytest.importorskip("gymnasium")
    from certgap.ppo import train_ppo

    cfg = PPOConfig(
        total_timesteps=4096,
        n_steps=1024,
        minibatch_size=64,
        policy_epochs=2,
        value_epochs=2,
    )
    result = train_ppo(env_id="CartPole-v1", seed=0, cfg=cfg, verbose=False)

    assert "metadata" in result
    assert "log" in result
    log = result["log"]
    expected_keys = set(per_update_log_keys())
    assert expected_keys <= set(log.keys())

    # All arrays same length, equal to n_updates.
    n_updates = result["metadata"]["n_updates"]
    for k, v in log.items():
        assert len(v) == n_updates, f"{k} has length {len(v)}, expected {n_updates}"

    # Final delta_J_k / harmful_k are NaN by construction.
    assert np.isnan(log["delta_J_k"][-1])
    assert np.isnan(log["harmful_k"][-1])
    # Earlier ones should be filled.
    if n_updates > 1:
        assert np.isfinite(log["delta_J_k"][0])

    # cert_gap_k = A_k - delta_hat_k (allow tiny float noise)
    cg = log["cert_gap_k"]
    a = log["A_k"]
    d = log["delta_hat_k"]
    assert np.allclose(cg, a - d, atol=1e-6, equal_nan=True)


@pytest.mark.slow
def test_tiny_run_with_heldout() -> None:
    pytest.importorskip("gymnasium")
    from certgap.ppo import train_ppo

    cfg = PPOConfig(
        total_timesteps=4096,
        n_steps=1024,
        minibatch_size=64,
        policy_epochs=2,
        value_epochs=2,
        heldout_batch_size=200,
    )
    result = train_ppo(
        env_id="CartPole-v1", seed=0, cfg=cfg, log_heldout=True, verbose=False
    )
    log = result["log"]
    assert "delta_hat_train_k" in log and "delta_hat_heldout_k" in log
    assert np.all(np.isfinite(log["delta_hat_train_k"]))
    assert np.all(np.isfinite(log["delta_hat_heldout_k"]))


@pytest.mark.slow
def test_tiny_run_with_nu_loss() -> None:
    """Verify PPO with nu_loss_weight > 0 completes and logs the ν-term."""
    pytest.importorskip("gymnasium")
    from certgap.ppo import train_ppo

    cfg = PPOConfig(
        total_timesteps=4096,
        n_steps=1024,
        minibatch_size=64,
        policy_epochs=2,
        value_epochs=2,
        nu_loss_weight=1.0,
    )
    result = train_ppo(env_id="CartPole-v1", seed=0, cfg=cfg, verbose=False)
    log = result["log"]
    # value_loss_nu should be populated with non-zero values
    assert "value_loss_nu" in log
    nu_vals = np.asarray(log["value_loss_nu"], dtype=float)
    assert np.all(np.isfinite(nu_vals))
    assert np.any(nu_vals > 0), "ν-loss term should be non-zero when enabled"
    # delta_hat_abs should also be logged
    assert "delta_hat_abs" in log
    assert np.all(np.isfinite(log["delta_hat_abs"]))
