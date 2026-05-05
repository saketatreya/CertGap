# The Certification Gap

Code and data for *The Bellman Residual Is Not a Policy-Improvement Proxy in
Actor--Critic Reinforcement Learning* (NeurIPS 2026 submission).

We show, theoretically and empirically, that the canonical Bellman residual
($\epsilon_u$) — the value-function loss minimized in nearly every PPO/SAC/TRPO
implementation — is uninformative as a per-update predictor of policy harm. The
exact identity $J(\pi_{k+1}) - J(\pi_k) = A_k - \hat\Delta_k$ exposes the
**start-state critic bias** $\hat\Delta_k = J(\pi_k) - \mathbb{E}_\nu V_{k+1}$ as
the operative covariate-shift quantity.

## Headline numbers

| Diagnostic | Pooled median AUROC | Wins (paired) |
|---|---|---|
| $-\epsilon_u$ (Bellman residual) | 0.46 | 5 / 262 |
| $-\hat\Delta_k$ (start-state bias) | **0.67** | **256 / 262** |

263 seeds across PPO, the PPO-MSE ablation, SAC, and TRPO; 8 environments;
~90,000 logged actor–critic updates. Paired Wilcoxon $p < 10^{-43}$.

## Repository layout

```
certgap/      Core PPO / SAC / TRPO implementation with paper-aligned logging
figures/      Figure scripts (one per paper figure) -> figures/out/
paper/        paper.tex, references.bib, neurips_2026.sty, checklist.tex
results/      Cached audit pickles (~90k updates, 263 seeds)
scripts/      run_all.py master runner + diagnostic utilities
tests/        Smoke tests
```

## Installation

```bash
make install
# equivalently:
#   python -m venv .venv
#   .venv/bin/pip install -e .
#   .venv/bin/pip install "gymnasium[mujoco]"
```

## Single-command reproduction

| Goal | Command |
|---|---|
| Run every experiment from scratch (~13h on 8-core CPU; idempotent) | `make experiments` |
| Regenerate every figure and table from cached `results/` pickles (~1 min) | `make figures` |
| Rebuild the paper PDF (`paper/paper.pdf`) | `make paper` |
| Smoke tests + tabular-identity sanity check | `make verify` |

`make experiments` skips runs whose pickle is already on disk, so it is safe to
interrupt and resume. Pass `WORKERS=N` to set the parallelism (default 6):
`make experiments WORKERS=8`.

The full pipeline, end-to-end:

```bash
make install
make experiments
make figures
make paper
```

## Paper

`paper/paper.tex` uses the official NeurIPS 2026 style (`paper/neurips_2026.sty`)
with anonymized author block and line numbers, and includes the NeurIPS Paper
Checklist (`paper/checklist.tex`) after the appendix. Build with `make paper`.

## License

MIT. See `LICENSE`.
