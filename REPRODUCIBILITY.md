# Reproducibility checklist

Following the NeurIPS 2026 reproducibility template. Each item references a
file path, function, command, or commit SHA in this repository.

## Code

- **Code is publicly accessible.**
  Repository: `https://github.com/<TBD>/CertGap`. License: MIT.
- **Code is documented.**
  Module docstrings on every package; reproduction recipes in `README.md`;
  data sheet in `DATASHEET.md`.
- **Code is structured.**
  See repository layout in `README.md`. Single Python package `certgap/`,
  one figure script per paper figure, runners separate from algorithms.
- **All dependencies are pinned.**
  `pyproject.toml` (Python ≥ 3.10; PyTorch, gymnasium\[mujoco\], numpy,
  scipy, matplotlib).

## Models

- **Architecture is fully specified.**
  `certgap/common/networks.py`: PolicyNet and ValueNet are both
  (64, 64)-Tanh MLPs with orthogonal initialization. Continuous-action
  policies use a Gaussian with state-independent log-std (paper Appendix E,
  matches Schulman et al. 2017).
- **Hyperparameters are fully specified.**
  `certgap/common/config.py:PPOConfig`. Defaults match paper Appendix E
  exactly: γ=0.99, GAE λ=0.95, batch=2048, minibatch=64, 10 policy-epochs,
  10 value-epochs, clip-ε=0.2, π-lr=3e-4, V-lr=1e-3, grad-clip=1.0,
  no entropy bonus, ε_u definition = "mse".
- **All training procedures are described.**
  `certgap/ppo.py:train_ppo`. Single function for all variants.
  Per-update log schema in `certgap/common/metrics.py`.

## Data

- **Data sources are documented.**
  All environments are public Gymnasium environments. See `DATASHEET.md`.
- **Data preprocessing is documented.**
  No preprocessing — the per-update logs are the released data.
  Computed by the training loop in real time.
- **Data splits are documented.**
  No traditional split. Per-environment results, plus pooled. Per-environment
  is reported with paired Wilcoxon statistics; pooled is reported with
  z-scored within-environment standardization.
- **Held-out evaluation is reproducible.**
  `--log-heldout` flag on `certgap/runners/run_one.py`; collects 500
  transitions per update from a separately-seeded evaluation environment.

## Experiments

- **Number of seeds is specified.**
  Main grid: 20 seeds per environment × 8 environments = 160 PPO runs.
  Factorials: 5 seeds per cell. ε_u variants: 6 seeds. Held-out: 20 seeds.
  Tabular: 8 seeds per family × 3 families = 24 runs.
- **Compute requirements are specified.**
  Total: ~78 CPU-hours across 540 PPO/SAC/TRPO training runs. At 6 parallel
  workers, ~13 wall-clock hours. Per-run wall time on a 2024 M-series Mac:
  CartPole ~4 min, LunarLander ~4 min, Hopper ~7 min, HalfCheetah ~8 min,
  Ant ~6 min, Walker2d ~7 min, Humanoid ~12 min.
- **Random seeds are seeded reproducibly.**
  `certgap/common/utils.py:set_global_seeds` seeds Python's `random`,
  numpy, and PyTorch generators. Per-run seeds are 0–19 for the main grid.
- **Statistical tests are specified.**
  AUROC, AUPRC, Pearson r (with Fisher-z CIs), Spearman r,
  paired Wilcoxon signed-rank (one-sided cg > -ε_u). Bootstrap 95% CIs
  (n=1000 resamples, RNG seeded at 20260429).
- **Evaluation metrics are described.**
  See paper §3 (audit setup) and `certgap/analysis/harm_prediction.py`.
- **Computing infrastructure is described.**
  CPU-only, single machine, multiprocess pool. 8-core M-series Mac. No GPU.

## Reproducibility

- **A single command produces all figures from cached data.**
  `make figures` regenerates 5 main + 3 appendix PDFs from `results/`
  pickles, plus the headline `figures/out/table1.tex` table.
- **A single command produces the headline table.**
  `python scripts/build_table1.py` writes `figures/out/table1.csv` and
  `figures/out/table1.tex`.
- **A single command runs the full pipeline.**
  `make experiments` runs every experiment in the paper. With existing
  pickles on disk, this is a no-op.
- **A single command rebuilds the paper.**
  `make paper` regenerates `paper/paper.pdf` from `paper/paper.tex`.
- **A smoke test passes.**
  `make verify` runs `pytest tests/` and the tabular-identity check.

## Assets

- **Citations are provided for all third-party assets.**
  - PyTorch, NumPy, SciPy, matplotlib (standard scientific stack).
  - Gymnasium / MuJoCo: Brockman et al. 2016 / Todorov et al. 2012.
  - PPO: Schulman et al. 2017. SAC: Haarnoja et al. 2018. TRPO:
    Schulman et al. 2015a. GAE: Schulman et al. 2015b.
- **License compatibility is verified.**
  All dependencies are MIT or BSD compatible with our MIT release.

## Limitations

- **Limitations of the work are described.**
  See paper §9 ("Limitations" paragraph). Specifically:
  - On-policy actor-critic only. Off-policy methods (SAC, TD3) require
    importance-weighted analogues of A_k that we do not test.
  - 8 environments; results may not generalize to large-scale benchmarks
    (Atari, robotics) or richer policy classes.
  - The Δ̂_k start-state estimator has high finite-sample variance on
    environments with few episodes per rollout (HalfCheetah: 2 episodes,
    Ant: 3 episodes).

## Ethical considerations

- **No human subjects data.**
- **No identifiable personal data.**
- **No safety-critical claims.**
  The paper explicitly establishes that the cert-gap *cannot* be used as a
  per-update safety controller (the diagnostic-controller boundary result).
  No safe-RL claims are made.
- **Compute footprint is modest.**
  ~78 CPU-hours total. CPU-only, no GPU. Reproducible on a laptop in
  ~13 wall-clock hours.

## Versioning

- Code: tagged `v1.0` at submission.
- Data: archived to Zenodo at submission as `bench-critic-v1`, with DOI.
- Dependencies: pinned in `pyproject.toml`.
