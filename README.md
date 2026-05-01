# The Certification Gap

Code and data for *The Bellman Residual Is Not a Policy-Improvement Proxy in Actor--Critic Reinforcement Learning* (NeurIPS 2026, under submission).

This repository demonstrates empirically and theoretically that the canonical Bellman residual ($\epsilon_u$) — the value-function loss used in nearly every PPO/SAC implementation — is uninformative as a per-update predictor of policy harm. We isolate the **start-state critic bias** ($\hat\Delta_k$) as the mechanistically active signal for policy improvement.

## Key Findings
- **Diagnostic Failure**: The absolute Bellman residual is near-chance at predicting policy harm (pooled median AUROC 0.46).
- **Start-State Bias**: The bias term $\hat\Delta_k = J(\pi_k) - \mathbb{E}_\nu V_{k+1}$ predicts harm at pooled median AUROC 0.68 and dominates $\epsilon_u$ on 235 of 237 seeds ( < 10^{-40}$).
- **The $ Paradox**: Surrogate advantage $ is often anti-predictive of actual improvement because the policy "chases" the critic's start-state bias.
- **Actionable Intervention**: A post-update validation gate based on $\hat\Delta_k$ successfully stabilizes training on high-dimensional tasks like Humanoid-v5.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
pip install "gymnasium[mujoco]"
```

## One-Line Reproducibility

**1. Regenerate all paper figures and tables from results:**
```bash
make figures
```
This produces all main and appendix PDFs in `figures/out/`, including the updated audit table.

**2. Run the full experimental suite from scratch:**
```bash
.venv/bin/python scripts/run_all.py --workers 4 --skip-if-exists
```
This command handles the main grid, hyperparameter factorials, baseline recomputes, and the gated intervention study.

## Repository Structure
- `certgap/`: Core implementation of PPO, SAC, and TRPO with paper-aligned logging.
- `results/`: The audit dataset (86k updates, 238 seeds).
- `figures/`: Scripts for every figure in the manuscript.
- `scripts/`: Masters runners and diagnostic utilities.
- `paper.pdf`: The submitted manuscript.

## License
MIT. See `LICENSE` for details.
