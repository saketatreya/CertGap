# Reproducibility checklist

Following the NeurIPS 2026 reproducibility template. Each item references a
file path, function, command, or commit SHA in this repository.

## Code

- [x] **Code is publicly accessible.**
  Repository: `https://github.com/<TBD>/CertGap`. License: MIT (`LICENSE`).
- [x] **Code is documented.**
  Module docstrings on every package; reproduction recipes in `README.md`;
  data sheet in `DATASHEET.md`.
- [x] **Code is structured.**
  See repository layout in `README.md`. Single Python package `certgap/`,
  one figure script per paper figure, runners separate from algorithms.
- [x] **All dependencies are pinned.**
  `pyproject.toml` (Python ≥ 3.10; PyTorch, gymnasium\[mujoco\], numpy,
  scipy, matplotlib).

## Models

- [x] **Architecture is fully specified.**
  `certgap/common/networks.py`: PolicyNet and ValueNet are both
  (64, 64)-Tanh MLPs with orthogonal initialization. Continuous-action
  policies use a Gaussian with state-independent log-std (paper Appendix H,
  matches Schulman et al. 2017).
- [x] **Hyperparameters are fully specified.**
  `certgap/common/config.py:PPOConfig`. Defaults match paper Appendix H
  exactly: γ=0.99, GAE λ=0.95, batch=2048, minibatch=64, 10 policy-epochs,
  10 value-epochs, clip-ε=0.2, π-lr=3e-4, V-lr=1e-3, grad-clip=1.0,
  no entropy bonus, ε_u definition = "mse".
- [x] **All training procedures are described.**
  `certgap/ppo.py:train_ppo`. ~280 lines, single function for all variants.
  Per-update log schema in `certgap/common/metrics.py:per_update_log_keys`.

## Data

- [x] **Data sources are documented.**
  All environments are public Gymnasium environments. See `DATASHEET.md`.
- [x] **Data preprocessing is documented.**
  No preprocessing — the per-update logs are the released data.
  Computed by the training loop in real time.
- [x] **Data splits are documented.**
  No traditional split. Per-environment results, plus pooled. Per-environment
  is reported with bootstrap CIs and paired Wilcoxon statistics; pooled is
  reported with z-scored within-environment standardization (paper §4.2).
- [x] **Held-out evaluation is reproducible.**
  `certgap/analysis/estimator_agreement.py`. Run with
  `--log-heldout` flag; collects 500 transitions per update from a
  separately-seeded evaluation environment.

## Experiments

- [x] **Number of seeds is specified.**
  Main grid: 20 seeds per environment × 8 environments = 160 PPO runs.
  Factorials: 5 seeds per cell. ε_u variants: 6 seeds. Held-out: 20 seeds.
  Tabular: 8 seeds per family × 3 families = 24 runs.
- [x] **Compute requirements are specified.**
  Total: ~78 CPU-hours across 457 PPO training runs. At 6 parallel workers,
  ~13 wall-clock hours. Per-run wall time on a 2024 M-series Mac:
  CartPole ~4 min, LunarLander ~4 min, Hopper ~7 min, HalfCheetah ~8 min,
  Ant ~6 min, Walker2d ~7 min, Humanoid ~12 min.
- [x] **Random seeds are seeded reproducibly.**
  `certgap/common/utils.py:set_global_seeds` seeds Python's `random`,
  numpy, and PyTorch generators. Per-run seeds are 0–19 for the main grid.
- [x] **Statistical tests are specified.**
  AUROC, AUPRC, Pearson r (with Fisher-z CIs), Spearman r,
  paired Wilcoxon signed-rank (one-sided cg > -ε_u). Bootstrap 95% CIs
  (n=1000 resamples, RNG seeded at 20260429).
- [x] **Evaluation metrics are described.**
  See paper §3 (empirical setup) and `certgap/analysis/harm_prediction.py`.
- [x] **Computing infrastructure is described.**
  CPU-only, single machine, multiprocess pool. 8-core M-series Mac. No GPU.

## Reproducibility

- [x] **A single command produces all figures from cached data.**
  `make figures` regenerates 5 main + 5 appendix PDFs from `results/`
  pickles. Total time: < 2 minutes.
- [x] **A single command produces the headline table.**
  `python -m scripts.build_table1` writes `figures/out/table1.csv` and
  `figures/out/table1.tex`.
- [x] **A single command runs the full pipeline.**
  `python scripts/run_all.py --workers 6 --skip-if-exists` runs every
  experiment in the paper. With existing pickles on disk, this is a no-op.
- [x] **A smoke test passes.**
  `make verify` runs a 50k-step CartPole training run and `pytest tests/`.
  Total time: ~5 minutes.

## Assets

- [x] **Citations are provided for all third-party assets.**
  - PyTorch, NumPy, SciPy, matplotlib (standard scientific stack).
  - Gymnasium / MuJoCo: Brockman et al. 2016 / Todorov et al. 2012.
  - PPO: Schulman et al. 2017.
  - GAE: Schulman et al. 2015b.
- [x] **License compatibility is verified.**
  All dependencies are MIT or BSD compatible with our MIT release.

## Limitations

- [x] **Limitations of the work are described.**
  See paper §5. Specifically:
  - On-policy actor-critic only. Off-policy methods (SAC, TD3) require
    importance-weighted analogues of A_k that we do not test.
  - 8 environments; results may not generalize to large-scale benchmarks
    (Atari, robotics) or richer policy classes.
  - The Δ̂_k start-state estimator has high finite-sample variance on
    environments with few episodes per rollout (HalfCheetah: 2 episodes,
    Ant: 3 episodes).

## Ethical considerations

- [x] **No human subjects data.**
- [x] **No identifiable personal data.**
- [x] **No safety-critical claims.**
  The paper explicitly establishes that the cert-gap *cannot* be used as a
  per-update safety controller (§4.4, the diagnostic-controller boundary
  result). No safe-RL claims are made.
- [x] **Compute footprint is modest.**
  ~78 CPU-hours total. CPU-only, no GPU. Reproducible on a laptop in
  ~13 wall-clock hours.

## Versioning

- Code: tagged `v1.0` at submission.
- Data: archived to Zenodo at submission as `bench-critic-v1`, with DOI.
- Dependencies: pinned in `pyproject.toml`.
