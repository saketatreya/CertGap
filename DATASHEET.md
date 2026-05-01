# Datasheet for the Certification-Gap Benchmark

Following the data-sheet template of Gebru et al. (2018), *"Datasheets for Datasets."*

## Motivation

**For what purpose was the dataset created?**
To test whether the canonical critic-quality scalar in actor-critic deep RL — the
Bellman residual `ε_u` — is informative about per-update policy harm, and to
provide a public benchmark for evaluating alternative critic-quality
diagnostics. Every PPO/SAC implementation we surveyed logs `ε_u` as the
value-function loss; this dataset is the rigorous test of whether that
practice tracks the quantity practitioners assume it does.

**Who created the dataset?**
The author, as part of a single-author submission to NeurIPS 2026 D&B.

**Who funded the creation of the dataset?**
No external funding. All compute is CPU on commodity hardware.

## Composition

**What do the instances of the dataset represent?**
Each instance is one PPO update during training of a Gymnasium environment.
For each update we record:

| Field | Type | Description |
|---|---|---|
| `update_idx` | int | Sequential update number within the run |
| `timesteps` | int | Cumulative environment steps at the update boundary |
| `A_k` | float | Surrogate advantage: importance-weighted GAE under π_{k+1} |
| `delta_hat_k` | float | Start-state critic bias: Ĵ(π_k) − E[V_{k+1}(s_0)] |
| `cert_gap_k` | float | A_k − delta_hat_k |
| `eps_u` | float | Mean squared TD error post-critic-update (paper-canonical) |
| `J_k` | float | Mean episode return on this rollout |
| `delta_J_k` | float | J_{k+1} − J_k (NaN on the final update) |
| `harmful_k` | int | 1 if delta_J_k < 0 else 0 (NaN on the final update) |
| `clip_frac` | float | PPO clip fraction on the first epoch |
| `policy_entropy` | float | Mean policy entropy after the update |
| `value_loss_pre/post` | float | Value MSE before/after the value-net update |
| `n_episodes` | int | Number of complete episodes in this rollout |

For a subset of runs additional fields are logged:
- `eps_u_mse`, `eps_u_sup`, `eps_u_p95` (ε_u under three definitions; eps_u_variants subset)
- `delta_hat_train_k`, `delta_hat_heldout_k` (residual-based Δ̂ on train and held-out batches; heldout_humanoid subset)

Tabular subset (separate, see below): one CSV row per softmax-NPG iteration
on a finite RandomMDP, with exact-DP-computed `A_k`, `delta_hat_k`,
`predicted_delta_J`, `delta_J_exact`, and `identity_error`.

**How many instances are there?**

```
Sweep                      runs   updates    schema
main                        160    24,500    base + per-env step budget
factorial_lunarlander       155    ~21,200   base
factorial_halfcheetah        71    ~10,600   base
factorial_hopper             45    ~6,750    base
heldout_humanoid             20    ~4,900    base + delta_hat_train_k, delta_hat_heldout_k
eps_u_variants                6    ~1,470    base + eps_u_{mse,sup,p95}
tabular                      24    ~2,400    exact-DP analogues (CSV)
TOTAL PPO                   457   ~70,707
```

**Does the dataset contain all possible instances or is it a sample of instances from a larger set?**
Sample. Each PPO run is one realization (env, seed, hyperparameter cell)
drawn from a large hypothetical space of training trajectories. The sampling
design is documented in §3 of the accompanying paper:

- 8 environments: 3 discrete-action (CartPole-v1, Acrobot-v1, LunarLander-v3),
  5 continuous-action MuJoCo (Hopper-v5, HalfCheetah-v5, Walker2d-v5,
  Ant-v5, Humanoid-v5).
- 20 seeds per environment for the main grid (seeds 0–19).
- Hyperparameter factorials: value-epochs × clip-ε grid, plus a
  value-learning-rate sweep on a single cell. 5 seeds per cell.
- Hyperparameters otherwise identical to paper Appendix H.

**Is there a label or target associated with each instance?**
Yes. The label is `harmful_k = 1[ΔJ_k < 0]`, where `ΔJ_k = J_{k+1} − J_k` is
the change in mean episode return between consecutive on-policy rollouts.
The label is missing (NaN) for the final update of each run because there is
no successor rollout.

**Is any information missing from individual instances?**
The final update of each run has no successor rollout, so `delta_J_k` and
`harmful_k` are NaN. All other fields are populated.

**Are relationships between individual instances made explicit?**
Yes. Instances within a run are temporally ordered by `update_idx`; the
`delta_J_k` of update k depends on `J_k` and `J_{k+1}`. Cross-run
relationships are environment-and-seed indexed.

**Are there recommended data splits?**
No traditional train/test split. The benchmark task is *post-hoc harm
prediction*: given features at update k, predict `harmful_k` (concurrent)
or `harmful_{k+1}` (lag-1, used to demonstrate the diagnostic-controller
boundary; the lag-1 task collapses to chance on every environment, see §5
of the paper).

For a method developer who wants to compare a new metric against the four
baselines reported here, we recommend: (a) report per-environment AUROC on
all 8 envs; (b) report pooled paired Wilcoxon against the Bellman residual
baseline; (c) report on the LunarLander 4×4 hyperparameter factorial to
demonstrate robustness.

**Are there any errors, sources of noise, or redundancies?**
- `delta_hat_k` uses the paper-formal start-state estimator
  Ĵ(π_k) − E[V_{k+1}(s_0)], which has high finite-sample variance because
  |S_0| (number of complete episodes per rollout) ranges from 2 to 26
  across environments. The residual-based estimator
  (mean Bellman residual / (1−γ)) has lower variance per update; we report
  both in the heldout_humanoid subset and use the formal estimator for
  the headline numbers because the residual estimator has higher per-update
  variance on out-of-rollout transitions (see appendix).
- Hyperparameters are identical to paper Appendix H, including no entropy
  bonus.

## Collection Process

**How was the data collected?**
By running a clean reimplementation of PPO (paper Appendix H configuration:
γ = 0.99, GAE λ = 0.95, batch 2048, minibatch 64, 10 policy-epochs, 10
value-epochs, clip-ε = 0.2, π-lr 3×10⁻⁴, V-lr 1×10⁻³, gradient clip 1.0,
Tanh-(64, 64) MLPs, no entropy bonus). Each PPO update logs the schema above.

The reference implementation is `certgap/ppo.py:train_ppo`. The launching
infrastructure is `certgap/runners/{run_one,sweep}.py`. The full reproduction
command set is in `scripts/run_all.py`.

**What mechanisms or procedures were used to collect the data?**
Single-machine multiprocess training on CPU. Total compute: ~78 CPU-hours.
At 6 workers, ≈13 wall-clock hours.

**Who was involved in the data collection process?**
The author. No human-subjects data is involved.

**Over what timeframe was the data collected?**
2026-04-29 (single execution).

**Were any ethical review processes conducted?**
Not applicable — synthetic RL training data only.

## Preprocessing / Cleaning / Labeling

**Was any preprocessing/cleaning/labeling of the data done?**
Yes:
- The label `harmful_k = 1[ΔJ_k < 0]` is computed by the training loop
  itself.
- All metric scalars are computed in 32-bit float and stored in numpy
  arrays. No further preprocessing.
- The held-out estimator-agreement subset adds a separately-seeded
  evaluation environment that collects 500 transitions per update,
  disjoint from the training rollout.

**Was the "raw" data saved in addition to the preprocessed/cleaned data?**
The raw per-update logs *are* the data. There is no further preprocessing
between collection and release; analysis modules
(`certgap/analysis/`) operate directly on the released pickles.

## Uses

**Has the dataset been used for any tasks already?**
The accompanying paper uses it to (a) evaluate four baseline harm
predictors (cert-gap, A_k alone, −Δ̂_k alone, −ε_u) on per-update harm
prediction; (b) measure hyperparameter robustness; (c) measure ε_u
definition robustness; (d) demonstrate the diagnostic-controller
boundary (lag-1 collapse to chance).

**Is there a repository that links to any or all papers or systems that use the dataset?**
The single accompanying paper. Repository:
`https://github.com/<TBD>/CertGap` (set on submission).

**What (other) tasks could the dataset be used for?**
- Evaluating new critic-quality diagnostics in actor-critic deep RL.
- Studying estimator variance of A_k and Δ̂_k on long PPO trajectories.
- Pre-training auxiliary harm predictors (e.g., learned ensembles).
- Studying the relationship between PPO clip fraction and harm.

**Is there anything about the composition of the dataset or the way it was collected and preprocessed/cleaned/labeled that might impact future uses?**
- The dataset is on-policy (PPO) only. Off-policy actor-critic methods
  (SAC, TD3) would need importance-corrected analogues of A_k.
- The eight environments are the standard Gymnasium suite. Real-world
  robotics or large-scale benchmarks (Atari, Crafter) are out of scope.
- All runs use a Gaussian policy with state-independent log-std on
  continuous-action environments; richer policy parameterizations are
  out of scope.

## Distribution

**Will the dataset be distributed to third parties outside of the entity on behalf of which the dataset was created?**
Yes. The dataset will be released publicly with the paper.

**How will the dataset be distributed?**
Two channels:
- The paper's GitHub repository, under `results/`.
- Zenodo, with a DOI assigned at submission for citability.

**When will the dataset be distributed?**
At paper submission (NeurIPS 2026 D&B).

**Will the dataset be distributed under a copyright or other intellectual property (IP) license, and/or under applicable terms of use (ToU)?**
- Code: MIT License (see `LICENSE`).
- Data: Creative Commons Attribution 4.0 International (CC-BY 4.0).

**Have any third parties imposed IP-based or other restrictions on the data associated with the instances?**
No.

## Maintenance

**Who will be supporting/hosting/maintaining the dataset?**
The author until publication; subsequently, the GitHub repository will be
maintained alongside the paper. Zenodo provides long-term archival.

**How can the owner/curator/manager of the dataset be contacted?**
Through the GitHub repository's issue tracker.

**Is there an erratum?**
Errata, if any, will be appended to the repository's `README.md`.

**Will the dataset be updated?**
The first release is versioned `bench-critic-v1`. Updates that change
schema will increment the major version; updates that add seeds or
environments will increment the minor version.

**If the dataset relates to people, are there applicable limits on the retention of the data associated with the instances?**
Not applicable — synthetic RL training data only.
