"""Randomly generated finite MDP with exact dynamic programming solvers.

The MDP is fully specified by (P, R, gamma, nu):
  - P[s, a, s'] = transition probability
  - R[s, a]     = expected reward
  - gamma       = discount factor
  - nu[s]       = initial state distribution

All exact computations (Q^mu, J(mu), d^mu, Delta_hat) use matrix algebra
on the stored MDP parameters.  No sampling is involved in the exact block.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


class RandomMDP:
    """A randomly generated finite MDP for exact identity verification."""

    def __init__(
        self,
        n_states: int = 64,
        n_actions: int = 4,
        gamma: float = 0.95,
        seed: int = 0,
    ) -> None:
        self.S = n_states
        self.A = n_actions
        self.gamma = gamma

        rng = np.random.default_rng(seed)

        # Transition kernel: Dirichlet(0.5) per (s, a) → sparse transitions
        self.P = np.zeros((n_states, n_actions, n_states))
        for s in range(n_states):
            for a in range(n_actions):
                self.P[s, a, :] = rng.dirichlet(np.full(n_states, 0.5))

        # Reward: uniform in [-1, 1]
        self.R = rng.uniform(-1.0, 1.0, size=(n_states, n_actions))

        # Initial state distribution: uniform
        self.nu = np.ones(n_states) / n_states

        # Current state for simulation
        self._state: int = 0
        self._rng = rng

    def reset(self) -> int:
        """Reset and return initial state sampled from nu."""
        self._state = self._rng.choice(self.S, p=self.nu)
        return self._state

    def step(self, action: int) -> tuple[int, float, bool]:
        """Take action, return (next_state, reward, done=False).

        Episodes never terminate in this MDP (infinite horizon, discounted).
        We use a fixed rollout length externally.
        """
        s = self._state
        r = self.R[s, action]
        s_next = self._rng.choice(self.S, p=self.P[s, action, :])
        self._state = s_next
        return s_next, float(r), False

    # ── Exact DP solvers ────────────────────────────────────────────────────

    def _policy_transition_matrix(self, policy: NDArray) -> NDArray:
        """P^mu[s, s'] = sum_a policy[s, a] * P[s, a, s'].

        Args:
            policy: shape (S, A), policy[s, a] = mu(a|s)

        Returns:
            shape (S, S)
        """
        # P is (S, A, S), policy is (S, A) → einsum over action
        return np.einsum("sa,sap->sp", policy, self.P)

    def _policy_reward_vector(self, policy: NDArray) -> NDArray:
        """r^mu[s] = sum_a policy[s, a] * R[s, a].

        Returns:
            shape (S,)
        """
        return np.einsum("sa,sa->s", policy, self.R)

    def solve_V(self, policy: NDArray) -> NDArray:
        """Solve V^mu = (I - gamma P^mu)^{-1} r^mu.

        Returns:
            shape (S,)
        """
        P_mu = self._policy_transition_matrix(policy)
        r_mu = self._policy_reward_vector(policy)
        I = np.eye(self.S)
        V = np.linalg.solve(I - self.gamma * P_mu, r_mu)
        return V

    def solve_Q(self, policy: NDArray) -> NDArray:
        """Solve Q^mu[s, a] = R[s, a] + gamma * sum_{s'} P[s,a,s'] V^mu[s'].

        Returns:
            shape (S, A)
        """
        V = self.solve_V(policy)
        # Q[s,a] = R[s,a] + gamma * P[s,a,:] @ V
        Q = self.R + self.gamma * np.einsum("sap,p->sa", self.P, V)
        return Q

    def solve_J(self, policy: NDArray) -> float:
        """J(mu) = nu^T V^mu."""
        V = self.solve_V(policy)
        return float(self.nu @ V)

    def solve_d(self, policy: NDArray) -> NDArray:
        """Discounted state visitation: d^mu = (1-gamma) nu^T (I - gamma P^mu)^{-1}.

        Returns:
            shape (S,), sums to 1.
        """
        P_mu = self._policy_transition_matrix(policy)
        I = np.eye(self.S)
        d = (1 - self.gamma) * self.nu @ np.linalg.inv(I - self.gamma * P_mu)
        return d

    def compute_delta_hat_exact(
        self,
        policy: NDArray,
        critic_V: NDArray,
    ) -> float:
        """Exact Δ̂_k = J(mu_k) - nu^T f_{k+1}.

        Here nu^T f_{k+1} = sum_s nu(s) f_{k+1}(s).

        Args:
            policy: mu_k, shape (S, A) (used for J(mu_k))
            critic_V: f_{k+1}(s), shape (S,)

        Returns:
            scalar Δ̂_k
        """
        J_mu = self.solve_J(policy)
        # nu^T f = sum_s nu(s) * f(s)
        nu_f = float(self.nu @ critic_V)
        return J_mu - nu_f

    def compute_A_k_exact(
        self,
        policy_new: NDArray,
        critic_V: NDArray,
    ) -> float:
        """Exact A_k = (d^{mu_{k+1}})^T (T^{mu_{k+1}} f - f) / (1-gamma).

        T^{mu_{k+1}} f[s] = sum_a mu_{k+1}(a|s) [ R[s,a] + gamma sum_{s'} P[s,a,s'] f(s') ]

        Args:
            policy_new: mu_{k+1}, shape (S, A)
            critic_V: f_{k+1}(s), shape (S,)

        Returns:
            scalar A_k
        """
        # T^{mu_{k+1}} f[s]
        # Q_f[s,a] = R[s,a] + gamma * P[s,a,:] @ critic_V
        Q_f = self.R + self.gamma * np.einsum("sap,p->sa", self.P, critic_V)
        # V_Tf[s] = sum_a policy_new[s,a] * Q_f[s,a]
        V_Tf = np.einsum("sa,sa->s", policy_new, Q_f)
        
        # Bellman residual: (T^mu f - f)[s]
        residual = V_Tf - critic_V  # (S,)

        # d^{mu_{k+1}}
        d = self.solve_d(policy_new)  # (S,)

        # A_k = sum_s d(s) * residual(s) / (1-gamma)
        A_k = np.sum(d * residual) / (1 - self.gamma)
        return float(A_k)

    def collect_transitions(
        self,
        policy: NDArray,
        n_transitions: int,
    ) -> tuple[NDArray, NDArray, NDArray, NDArray]:
        """Collect transitions by simulating the MDP.

        Returns:
            states: (N,) int
            actions: (N,) int
            rewards: (N,) float
            next_states: (N,) int
        """
        states, actions, rewards, next_states = [], [], [], []
        s = self.reset()
        for _ in range(n_transitions):
            a = self._rng.choice(self.A, p=policy[s])
            s_next, r, _ = self.step(a)
            states.append(s)
            actions.append(a)
            rewards.append(r)
            next_states.append(s_next)
            s = s_next
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
        )
