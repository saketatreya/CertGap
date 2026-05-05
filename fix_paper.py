import re

with open('paper/paper.tex', 'r') as f:
    content = f.read()

# --- A1. Algebra error in proof of Proposition 1 ---
content = content.replace(
    r"""(1{-}\gamma) A_k
= \nu^\top (I - \gamma P^{\pi_{k+1}})^{-1} \big(r^{\pi_{k+1}} - (I - \gamma P^{\pi_{k+1}}) V_{k+1}\big)
= J(\pi_{k+1}) - \mathbb{E}_\nu V_{k+1}.""",
    r"""A_k
= \nu^\top (I - \gamma P^{\pi_{k+1}})^{-1} \big(r^{\pi_{k+1}} - (I - \gamma P^{\pi_{k+1}}) V_{k+1}\big)
= J(\pi_{k+1}) - \mathbb{E}_\nu V_{k+1}."""
)

# --- A2. Informal vs. formal mismatch in Theorem ---
content = content.replace(
    r"Rollout-only diagnostics cannot prospectively certify improvement.",
    r"No rollout-only scalar diagnostic can prospectively certify improvement under typical conditions of exploration and finite-sample or function-approximation fallibility."
)

# --- A3. Proposition 2 statement mixes scalarizations ---
content = content.replace(
    r"yet for which $\tilde\rho_k \le 1 - \gamma < 1$ for all $k$.",
    r"for $\epsilon_{u,k}$ evaluated in any positively-homogeneous norm, yet for which $\tilde\rho_k \le 1 - \gamma < 1$ for all $k$."
)

# --- A4. Proposition 2 footnote on constants ---
content = content.replace(
    r"\footnote{The constants $c_1{=}2$, $c_2{=}1$, $c_3{=}1$ correspond to the standard CPI mixture-policy bound combined with an approximate-critic correction \citep{kakade2002approximately, schulman2015trust}; the dimensional argument below is independent of the exact constants.}",
    r""
)

content = content.replace(
    r"\Psi(\alpha) \;=\; 2 \alpha A_k \;-\; \alpha^2 \frac{C_k^2}{1-\gamma} \;-\; \alpha \frac{C_k \,\epsilon_u}{1-\gamma},",
    r"\Psi(\alpha) \;=\; c_1 \alpha A_k \;-\; \alpha^2 \frac{c_2 C_k^2}{1-\gamma} \;-\; \alpha \frac{c_3 C_k \,\epsilon_u}{1-\gamma},"
)
content = content.replace(
    r"since $\Psi$ is a downward parabola with $\Psi(0)=0$, its unconstrained maximum on $\alpha > 0$ is positive iff $\Psi'(0) > 0$, i.e.\ $2(1-\gamma)A_k > C_k \epsilon_u$. Equivalently, the bound certifies progress at some step size if and only if the dimensionless ratio",
    r"Since $\Psi$ is a downward parabola with $\Psi(0)=0$, its unconstrained maximum on $\alpha > 0$ is positive iff $\Psi'(0) > 0$, i.e.\ $c_1(1-\gamma)A_k > c_3 C_k \epsilon_u$. Equivalently, substituting the standard constants $c_1{=}2, c_2{=}1, c_3{=}1$ \citep{kakade2002approximately, schulman2015trust}, the bound certifies progress at some step size if and only if the dimensionless ratio"
)

# --- B1, B4, E2, E12. Abstract fixes ---
content = content.replace(
    r"where $\hat\Delta_k := J(\pi_k) - \mathbb{E}_{s \sim \nu}[V_{k+1}(s)]$ captures critic error under the start-state distribution; the Bellman residual does not appear in the improvement decomposition.",
    r"where $\hat\Delta_k := J(\pi_k) - \mathbb{E}_{s \sim \nu}[V_{k+1}(s)]$ measures the start-state bias of the post-update critic. The Bellman residual does not appear in the improvement decomposition."
)
content = content.replace(
    r"across $\sim$90{,}000 updates from PPO, PPO-MSE, SAC, and TRPO on eight environments (263 seeds total)",
    r"across $\sim$90{,}000 updates spanning four configurations (PPO, PPO-MSE, SAC, TRPO) on up to eight environments (263 seeds total)"
)
content = content.replace(
    r"(0.67) and dominates $-\widehat\epsilon_u$ on 256 of 262 paired seeds ($p < 10^{-44}$)",
    r"(0.68) and dominates $-\widehat\epsilon_u$ on 256 of 262 paired seeds ($p < 10^{-43}$)"
)

# --- D1, D3, B1, E3. Intro fixes ---
# I will do targeted replacements for the intro.
content = content.replace(
    r"""This is not a bound and not an explanation. It is the ground-truth decomposition of policy improvement, and it contains no Bellman residual term. It reveals why the standard monitor $\epsilon_u$ is fundamentally misaligned with the objective. While it may seem surprising that such a gap has persisted in the literature, we find that $\epsilon_u$ and $\hat\Delta_k$ are sufficiently correlated during the high-entropy phase of early training that the mismatch manifests clearly only as the policy approaches convergence --- exactly when monitoring is most critical. This mismatch leads to an ``$A_k$ paradox'' where the optimization target itself becomes misleading (pooled median AUROC 0.39), a finding that holds even in environments with dense start-state sampling.

\paragraph{$\epsilon_u$ is the wrong scale, not just a lossy proxy.}
The identity already shows $\epsilon_u$ is the wrong \emph{quantity}; we further show it is the wrong \emph{scale}. The standard improvement bound certifies progress for some step size if and only if the dimensionless ratio $\tilde\rho_k := 2(1-\gamma) A_k / (C_k \epsilon_u)$ exceeds one (\Cref{sec:theory}). The relevant tolerance is $\epsilon_u / A_k$, not $\epsilon_u$ in absolute terms; as the policy improves, $A_k \to 0$, and any fixed absolute $\epsilon^{\text{fixed}}$ leaves $\tilde\rho_k$ vacuous. \Cref{prop:no-go} formalizes this dimensional mismatch.

\paragraph{Empirical confirmation across $\sim$90{,}000 updates.}
The structural prediction is unambiguous: $\epsilon_u$ should not track policy improvement, and $\hat\Delta_k$ should. We audit $263$ seeds across PPO, the PPO-MSE ablation, SAC, and TRPO on eight environments. Because $\epsilon_u$ measures critic error correctly, but on the wrong distribution, it is \emph{anti-predictive} of policy harm (pooled median AUROC $0.47$, below chance --- informative but with the wrong sign). Because $\hat\Delta_k$ reflects the actual covariate shift, it pools to $0.67$ and dominates $\epsilon_u$ on $\mathbf{262\,\text{of}\,267}$ paired seeds (paired Wilcoxon $p < 10^{-44}$). The dominance is uniform across all four configurations, robust to a $\sim 200$-cell hyperparameter factorial, and survives on fully independent held-out rollouts (AUROC $0.61$).""",
    r"""This is not a bound and not an explanation. It is the ground-truth decomposition of policy improvement, and it reveals why the standard monitor $\epsilon_u$ is fundamentally misaligned with the objective: the Bellman residual does not appear in the improvement decomposition at all. While classical bounds introduce $\epsilon_u$ as scaffolding, the exact algebraic resolution shows the start-state bias $\hat\Delta_k$ is the operative quantity. Furthermore, as we show in \Cref{sec:theory}, $\epsilon_u$ is also at the wrong \emph{scale}: the relevant tolerance is $\epsilon_u / A_k$, so any fixed absolute tolerance eventually fails to certify even genuine improvement as $A_k \to 0$.

\paragraph{Empirical confirmation across $\sim$90{,}000 updates.}
The structural prediction is unambiguous: $\epsilon_u$ should not track policy improvement, and $\hat\Delta_k$ should. We audit $263$ seeds across PPO, the PPO-MSE ablation, SAC, and TRPO on up to eight environments. Because $\epsilon_u$ measures critic error correctly, but on the wrong distribution, it is \emph{anti-predictive} of policy harm (pooled median AUROC $0.47$, below chance --- informative but with the wrong sign). Because $\hat\Delta_k$ reflects the actual covariate shift, it pools to $0.68$ and dominates $\epsilon_u$ on $\mathbf{256\,\text{of}\,262}$ paired seeds (paired Wilcoxon $p < 10^{-43}$). The dominance is uniform across all four configurations, robust across a $170$-run hyperparameter factorial, and survives on fully independent held-out rollouts (AUROC $0.61$). This mismatch leads to an ``$A_k$ paradox'' where the surrogate advantage $A_k$ optimized by PPO can become anti-predictive in complex environments (pooled median AUROC 0.39)."""
)

# --- D2. Expand related work ---
content = content.replace(
    r"""The complementary off-policy distribution-shift literature \citep{islam2019off,prudencio2023survey} addresses a different mismatch (behavior vs.\ target policy); the gap we identify persists even strictly on-policy.""",
    r"""The complementary off-policy distribution-shift and offline RL literature \citep{islam2019off, nachum2019dualdice, voloshin2021empirical, prudencio2023survey} addresses a related but distinct mismatch (behavior vs.\ target policy) often via stationary distribution corrections; however, the gap we identify persists even strictly on-policy."""
)

# --- V_k vs V_{k+1} Discrepancy ---
content = content.replace(
    r"""\paragraph{Practical estimators.}
In a deep RL implementation, the population quantities of (\ref{eq:Ak-dhat}) are not directly observable. The estimators we use are:
\begin{align}
\label{eq:Ak-est}""",
    r"""\paragraph{Practical estimators.}
In a deep RL implementation, the population quantities of \Cref{eq:Ak-dhat} are not directly observable. Note that while the exact population identity (\Cref{eq:identity}) requires the post-update critic $V_{k+1}$, standard deep RL implementations compute the surrogate $\widehat{A}_k$ via GAE using the pre-update critic $V_k$. We evaluate the standard estimator to reflect the actual optimization signal utilized in practice. The estimators we use are:
\begin{align}
\label{eq:Ak-est}"""
)

# --- Set vs Multiset notation ---
content = content.replace(
    r"""\widehat{\hat\Delta}_k
  &\;:=\; \widehat{J}(\pi_k) \;-\; \tfrac{1}{|\mathcal{S}_0|}\sum_{s_0 \in \mathcal{S}_0} V_{k+1}(s_0), \\[0.2em]""",
    r"""\widehat{\hat\Delta}_k
  &\;:=\; \widehat{J}(\pi_k) \;-\; \tfrac{1}{N_{\text{eps}}}\sum_{i=1}^{N_{\text{eps}}} V_{k+1}(s_0^{(i)}), \\[0.2em]"""
)
content = content.replace(
    r"""and $\mathcal{S}_0$ is the set of start states sampled in the rollout. We adopt the start-state form (\ref{eq:dhat-est}) for $\widehat{\hat\Delta}_k$ throughout; the alternative residual-based estimator (\ref{eq:delta-hat-res})""",
    r"""and $N_{\text{eps}}$ is the number of episodes sampled in the rollout, with $s_0^{(i)}$ being the start state of the $i$-th episode. We adopt the start-state form \Cref{eq:dhat-est} for $\widehat{\hat\Delta}_k$ throughout; the alternative residual-based estimator \Cref{eq:delta-hat-res}"""
)

# --- B6. SAC seed count ---
content = content.replace(
    r"Hopper-v5, HalfCheetah-v5, Walker2d-v5, Ant-v5, Humanoid-v5, $6$--$10$ seeds each ($46$ runs).",
    r"Hopper-v5, HalfCheetah-v5, Walker2d-v5, Ant-v5, Humanoid-v5, $6$--$10$ seeds each ($45$ runs)."
)

# --- B7. TRPO LunarLander ---
content = content.replace(
    r"TRPO \citep{schulman2015trust} on LunarLander-v3 and Hopper-v5 ($10$ seeds each).",
    r"TRPO \citep{schulman2015trust} on LunarLander-v3 and Hopper-v5 ($10$ seeds each; 1 LunarLander seed stalled yielding 9 valid seeds)."
)

# --- C1. PPO-NU is missing from Section 4 ---
content = content.replace(
    r"TRPO \citep{schulman2015trust} on LunarLander-v3 and Hopper-v5 ($10$ seeds each; 1 LunarLander seed stalled yielding 9 valid seeds).",
    r"TRPO \citep{schulman2015trust} on LunarLander-v3 and Hopper-v5 ($10$ seeds each; 1 LunarLander seed stalled yielding 9 valid seeds). PPO-NU: an ablation adding an explicit $\nu$-weighted MSE loss term, on Hopper-v5 and Humanoid-v5 ($15$ seeds total)."
)
# And in Appendix J
content = content.replace(
    r"\emph{PPO-MSE} replaces the clipped value loss with raw MSE; otherwise identical.",
    r"\emph{PPO-MSE} replaces the clipped value loss with raw MSE; otherwise identical. \emph{PPO-NU} adds a start-state MSE loss term with coefficient $1.0$, sampled from a separate start-state buffer."
)

# --- B3, B4, B5, B1. Results numbers fixes ---
content = content.replace(
    r"pools to $0.67$ and beats $\widehat\epsilon_u$ on 256 of 262 paired seeds (paired Wilcoxon $p < 10^{-43}$). Every per-algorithm slice shows the same direction: PPO ($158/158$, $p < 10^{-27}$), the PPO-MSE ablation ($39/40$, $p < 10^{-12}$), SAC ($40/45$, $p < 10^{-8}$), and TRPO ($19/19$, $p < 10^{-6}$). While the direction is uniform, we note that the SAC signal is generally weaker than PPO (pooled $0.58$ vs $0.67$),",
    r"pools to $0.68$ and beats $\widehat\epsilon_u$ on 256 of 262 paired seeds (paired Wilcoxon $p < 10^{-43}$). Every per-algorithm slice shows the same direction: PPO ($158/158$, $p < 10^{-27}$), the PPO-MSE ablation ($39/40$, $p < 10^{-12}$), SAC ($40/45$, $p < 10^{-8}$), and TRPO ($19/19$, $p < 10^{-6}$). While the direction is uniform, we note that the SAC signal is generally weaker than PPO (pooled $0.59$ vs $0.68$),"
)
content = content.replace(
    r"\Cref{tab:per-env-auroc} (PPO-MSE rows) shows the same pattern: $-\widehat\epsilon_u$ at chance ($0.50$ pooled), $-\widehat{\hat\Delta}_k$ at $0.67$.",
    r"\Cref{tab:per-env-auroc} (PPO-MSE rows) shows the same pattern: $-\widehat\epsilon_u$ at chance ($0.48$ pooled), $-\widehat{\hat\Delta}_k$ at $0.67$."
)

# --- E1. Section heading "period." ---
content = content.replace(
    r"\paragraph{The PPO-MSE ablation: $\epsilon_u$ fails on the rollout distribution, period.}",
    r"\paragraph{The PPO-MSE ablation: $\epsilon_u$ fails on the rollout distribution.}"
)

# --- D3. The A_k paradox ---
content = content.replace(
    r"\paragraph{The $A_k$ paradox: the mechanism of optimization-signal corruption.}",
    r"\subsection{The $A_k$ paradox: the mechanism of optimization-signal corruption}"
)

# --- My Report C: Clarifying the Mechanism of A_k Paradox ---
content = content.replace(
    r"""The mechanism of the paradox is visible in the coupling of $A_k$ and $\hat\Delta_k$: $A_k$ is largest precisely when $\hat\Delta_k$ is large (\Cref{fig:mechanism-intervention}, Panel B). This implies a failure mode where the critic overfits to mid-trajectory states, causing it to badly underestimate start-state values. The resulting surrogate $A_k$ is then inflated, driving the policy to make an aggressive but uninformed step. One might suspect""",
    r"""The mechanism of the paradox is visible in the coupling of $A_k$ and $\hat\Delta_k$: $A_k$ is largest precisely when $\hat\Delta_k$ is large (\Cref{fig:mechanism-intervention}, Panel B). We hypothesize a failure mode where the critic overfits to mid-trajectory states, causing it to badly underestimate start-state values. Because GAE accumulates temporal difference errors, underestimating $V(s_0)$ while accurately estimating mid-trajectory $V(s_t)$ artificially inflates the temporal difference errors early in the rollout. The resulting surrogate $A_k$ is then inflated for actions that merely survive the start state, driving the policy to make an aggressive but uninformed step. One might suspect"""
)

# --- D4. Comparison to alternative diagnostics ---
content = content.replace(
    r"""\paragraph{Comparison to alternative diagnostics.}
We evaluate $\hat\Delta_k$ against standard PPO diagnostics on Humanoid-v5 to confirm its distinctive informativeness. On a representative single-seed audit (seed 2026, literal RNG seed), $-\widehat{\hat\Delta}_k$ achieves AUROC $0.86$, while the standard rollout value-loss $\epsilon_u$ ($0.54$), a held-out evaluation MSE ($0.60$, evaluated on an independent rollout from a separately-seeded environment), policy entropy ($0.67$), and grad norm ($0.55$) all provide substantially weaker predictive signals. While the median $-\widehat{\hat\Delta}_k$ AUROC across 20 seeds is lower ($0.75$, \Cref{tab:per-env-auroc}), its dominance over rollout-fit residuals remains uniform. This suggests the start-state bias is not merely tracking general training stability or convergence rate, but is uniquely sensitive to the distribution mismatch that governs policy improvement.""",
    r"""\paragraph{Comparison to alternative diagnostics.}
We evaluate $\hat\Delta_k$ against standard PPO diagnostics on Humanoid-v5 to confirm its distinctive informativeness. The median AUROC across 20 seeds for $-\widehat{\hat\Delta}_k$ is $0.75$ (\Cref{tab:per-env-auroc}), while the standard rollout value-loss $\epsilon_u$ (median $0.51$), a held-out evaluation MSE (median $0.61$, evaluated on an independent rollout), policy entropy (median $0.58$), and grad norm (median $0.52$) all provide substantially weaker predictive signals across the seeds. This suggests the start-state bias is not merely tracking general training stability or convergence rate, but is uniquely sensitive to the distribution mismatch that governs policy improvement."""
)

# --- E10. Table 2 caption ---
content = content.replace(
    r"caption{\textbf{Per-environment $A_k$ AUROC and Spearman correlation with $\hat\Delta_k$ on PPO updates.}",
    r"caption{\textbf{Per-environment $A_k$ AUROC and Spearman correlation with $\hat\Delta_k$ on PPO updates.} Positive Spearman $\rho$ indicates $A_k$ and $\hat\Delta_k$ co-rise. "
)

# --- D5. Section 6 renaming ---
content = content.replace(
    r"\section{Prospective limits and a retrospective gate}",
    r"\section{Why the mismatch is not closeable by simple reweighting}"
)

# --- E11. Theorem* informal ---
content = content.replace(
    r"\begin{theorem*}[Prospective limit; informal]",
    r"\begin{theorem*}[Prospective limit; informal, see \Cref{app:prospective-proof} for formal statement and proof]"
)

# --- A5. Estimator dimensional check ---
content = content.replace(
    r"The estimators agree in expectation (slope near $1$)",
    r"The estimators agree in expectation (since $\mathbb{E}_{d^{\pi_k}}[T^{\pi_k}V - V] / (1-\gamma) = J(\pi_k) - \mathbb{E}_\nu V$, which implies slope near $1$)"
)

# --- Fix (\ref{eq:...}) to \Cref{eq:...} in text ---
content = content.replace(
    r"(\ref{eq:Ak-dhat})",
    r"\Cref{eq:Ak-dhat}"
)
content = content.replace(
    r"(\ref{eq:dhat-est})",
    r"\Cref{eq:dhat-est}"
)
content = content.replace(
    r"(\ref{eq:delta-hat-res})",
    r"\Cref{eq:delta-hat-res}"
)

with open('paper/paper.tex', 'w') as f:
    f.write(content)

