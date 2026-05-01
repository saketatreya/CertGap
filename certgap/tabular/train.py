"""Bridge experiment: REINFORCE + exact identity verification.

At each iteration k:
  1. Collect N transitions under mu_k (simulation)
  2. Train critic f_{k+1} via MSE Bellman loss (50 SGD steps)
  3. EXACT DP: compute J(mu_k), J(mu_{k+1}), Delta_hat, A_k, identity error
  4. REINFORCE actor update
  5. Log everything
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch
from torch import optim

from certgap.tabular.mdp import RandomMDP
from certgap.tabular.networks import BridgeActor, BridgeCritic


def train_bridge(
    mdp_seed: int = 0,
    n_states: int = 64,
    n_actions: int = 4,
    n_iterations: int = 200,
    n_transitions: int = 2000,
    critic_sgd_steps: int = 50,
    critic_hidden: int = 16,
    actor_lr: float = 1e-3,
    critic_lr: float = 1e-3,
    gamma: float = 0.95,
    output_dir: str = "results/bridge",
    skip_if_exists: bool = False,
) -> dict:
    """Run one Bridge experiment seed.

    Returns:
        dict with all logged quantities (lists of length n_iterations).
    """
    out = Path(output_dir)
    csv_path = out / f"bridge_seed{mdp_seed}_h{critic_hidden}.csv"
    if skip_if_exists and csv_path.exists():
        print(f"  [skip] {csv_path} exists.")
        return {}

    mdp = RandomMDP(n_states=n_states, n_actions=n_actions, gamma=gamma, seed=mdp_seed)

    actor = BridgeActor(mdp.S, mdp.A, hidden=32)
    critic = BridgeCritic(mdp.S, hidden=critic_hidden)

    actor_opt = optim.Adam(actor.parameters(), lr=actor_lr)
    critic_opt = optim.Adam(critic.parameters(), lr=critic_lr)

    eye = torch.eye(mdp.S)  # one-hot encoding matrix

    log = {
        "iteration": [],
        "J_k": [],
        "J_k1": [],
        "delta_J_exact": [],
        "delta_hat_k": [],
        "A_k": [],
        "predicted_delta_J": [],
        "identity_error": [],
        "fa_error_linf": [],
        "fa_error_l2": [],
    }

    for k in range(n_iterations):
        # ── 1. Get current policy table ──────────────────────────────────
        policy_k = actor.get_policy_table().numpy()  # (S, A)

        # ── 2. Collect transitions under mu_k ────────────────────────────
        states, actions, rewards, next_states = mdp.collect_transitions(
            policy_k, n_transitions
        )

        # ── 3. Train critic f_{k+1} via MSE Bellman loss ────────────────
        states_oh = eye[states]           # (N, S)
        next_states_oh = eye[next_states] # (N, S)
        rewards_t = torch.tensor(rewards, dtype=torch.float32)
        policy_k_t = torch.tensor(policy_k, dtype=torch.float32)

        for _ in range(critic_sgd_steps):
            v_s = critic(states_oh).squeeze(-1)  # (N,)

            with torch.no_grad():
                v_next = critic(next_states_oh).squeeze(-1)  # (N,)
                target = rewards_t + gamma * v_next  # (N,)

            loss = torch.nn.functional.mse_loss(v_s, target)
            critic_opt.zero_grad()
            loss.backward()
            critic_opt.step()

        # ── 4. Extract critic table for exact computation ────────────────
        critic_V = critic.get_V_table().numpy()  # (S,)

        # ── 5. Exact DP block ────────────────────────────────────────────
        J_k = mdp.solve_J(policy_k)
        delta_hat_k = mdp.compute_delta_hat_exact(policy_k, critic_V)

        # FA error: ||f - V^mu||
        V_exact = mdp.solve_V(policy_k)
        fa_error_linf = float(np.max(np.abs(critic_V - V_exact)))
        fa_error_l2 = float(np.sqrt(np.mean((critic_V - V_exact) ** 2)))

        # ── 6. REINFORCE actor update ────────────────────────────────────
        states_oh_actor = eye[states]
        logits = actor(states_oh_actor)
        dist = torch.distributions.Categorical(logits=logits)
        actions_t = torch.tensor(actions, dtype=torch.long)
        log_probs = dist.log_prob(actions_t)

        with torch.no_grad():
            v_s_eval = critic(states_oh_actor).squeeze(-1)
            v_next_eval = critic(eye[next_states]).squeeze(-1)
            # TD advantage: r + gamma*V(s') - V(s)
            advantages = torch.tensor(rewards, dtype=torch.float32) + gamma * v_next_eval - v_s_eval

        policy_loss = -(log_probs * advantages).mean()
        actor_opt.zero_grad()
        policy_loss.backward()
        actor_opt.step()

        # ── 7. Compute J(mu_{k+1}) and A_k exactly ──────────────────────
        policy_k1 = actor.get_policy_table().numpy()
        J_k1 = mdp.solve_J(policy_k1)
        A_k = mdp.compute_A_k_exact(policy_k1, critic_V)

        delta_J_exact = J_k1 - J_k
        predicted_delta_J = A_k - delta_hat_k
        identity_error = abs(delta_J_exact - predicted_delta_J)

        # ── 8. Log ──────────────────────────────────────────────────────
        log["iteration"].append(k)
        log["J_k"].append(J_k)
        log["J_k1"].append(J_k1)
        log["delta_J_exact"].append(delta_J_exact)
        log["delta_hat_k"].append(delta_hat_k)
        log["A_k"].append(A_k)
        log["predicted_delta_J"].append(predicted_delta_J)
        log["identity_error"].append(identity_error)
        log["fa_error_linf"].append(fa_error_linf)
        log["fa_error_l2"].append(fa_error_l2)

        if k % 20 == 0 or k == n_iterations - 1:
            print(
                f"  [{k:3d}/{n_iterations}] J={J_k:.4f}  ΔJ={delta_J_exact:+.4f}"
                f"  Δ̂={delta_hat_k:+.4f}  A_k={A_k:+.4f}"
                f"  err={identity_error:.2e}  FA={fa_error_linf:.4f}"
            )

    # ── Save CSV ─────────────────────────────────────────────────────────
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / f"bridge_seed{mdp_seed}_h{critic_hidden}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(log.keys()))
        writer.writeheader()
        for i in range(n_iterations):
            writer.writerow({k: log[k][i] for k in log})

    print(f"  Saved: {csv_path}")
    return log
