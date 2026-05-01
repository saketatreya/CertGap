"""Cheapest possible CI: every public module imports."""

from __future__ import annotations


def test_imports() -> None:
    import certgap
    import certgap.common.config
    import certgap.common.metrics
    import certgap.common.networks
    import certgap.common.rollout
    import certgap.common.utils
    import certgap.common.gae
    import certgap.tabular.mdp
    import certgap.tabular.networks
    import certgap.tabular.train
    import certgap.geometry.jacobian
    import certgap.geometry.certified_ratio
    import certgap.ppo
    import certgap.sac
    import certgap.trpo
    import certgap.runners.run_one
    import certgap.runners.sweep
    import certgap.analysis.harm_prediction
    import certgap.analysis.correlations
    import certgap.analysis.estimator_agreement
    import certgap.analysis.factorial
    import certgap.analysis.audit


def test_per_update_log_keys_complete() -> None:
    from certgap.common.metrics import per_update_log_keys

    base = per_update_log_keys()
    with_heldout = per_update_log_keys(log_heldout=True)
    with_eps = per_update_log_keys(log_eps_u_variants=True)
    assert set(base) <= set(with_heldout)
    assert set(base) <= set(with_eps)
    # paper headline keys must be present
    for key in ("A_k", "delta_hat_k", "cert_gap_k", "eps_u", "delta_J_k", "harmful_k"):
        assert key in base
