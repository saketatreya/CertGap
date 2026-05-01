"""Preliminary analysis of the main 8-env × 20-seed grid + tabular.

Reproduces paper.pdf Table 1 from the new pickles, plus pooled Pearson r,
plus tabular identity max/median residual, plus the partial factorial
numbers if data is available.
"""

from __future__ import annotations

import csv
import pickle
from pathlib import Path

import numpy as np

from certgap.analysis.correlations import paired_wilcoxon_one_sided_greater, pearson
from certgap.analysis.harm_prediction import auprc, auroc

ENV_ORDER = [
    "LunarLander-v3", "CartPole-v1", "Acrobot-v1",
    "Hopper-v5", "HalfCheetah-v5", "Walker2d-v5", "Ant-v5", "Humanoid-v5",
]

# Paper.pdf Table 1 numbers (cg AUROC, ε_u AUROC, base rate).
PAPER_TABLE1 = {
    "LunarLander-v3":  (0.86, 0.46, 0.48),
    "CartPole-v1":     (0.63, 0.42, 0.25),
    "Acrobot-v1":      (0.94, 0.36, 0.43),
    "Hopper-v5":       (0.80, 0.52, 0.33),
    "HalfCheetah-v5":  (0.76, 0.57, 0.43),
    "Walker2d-v5":     (0.71, 0.48, 0.44),
    "Ant-v5":          (0.72, 0.48, 0.48),
    "Humanoid-v5":     (0.96, 0.53, 0.46),
}


def load_seed_logs(env: str) -> list[dict]:
    base = Path("results/main") / env
    out = []
    for pkl in sorted(base.glob("seed_*.pkl")):
        with open(pkl, "rb") as f:
            out.append(pickle.load(f))
    return out


def per_seed_metric_harm(log: dict, metric_key: str, *, sign: int = 1) -> tuple[np.ndarray, np.ndarray]:
    m = sign * np.asarray(log[metric_key], dtype=float)
    h = np.asarray(log["harmful_k"], dtype=float)
    return m, h


def per_seed_auroc(seed_logs: list[dict], key: str, sign: int = 1) -> np.ndarray:
    out = []
    for sl in seed_logs:
        m, h = per_seed_metric_harm(sl["log"], key, sign=sign)
        out.append(auroc(m, h))
    return np.asarray(out, dtype=float)


def per_seed_auprc(seed_logs: list[dict], key: str, sign: int = 1) -> np.ndarray:
    out = []
    for sl in seed_logs:
        m, h = per_seed_metric_harm(sl["log"], key, sign=sign)
        # auprc uses -metric internally to predict harm; sign flips back.
        out.append(auprc(m * sign, h))
    return np.asarray(out, dtype=float)


def env_base_rate(seed_logs: list[dict]) -> float:
    rates = []
    for sl in seed_logs:
        h = np.asarray(sl["log"]["harmful_k"], dtype=float)
        h = h[np.isfinite(h)]
        if h.size:
            rates.append(float(h.mean()))
    return float(np.mean(rates)) if rates else float("nan")


def pooled_pearson(seed_logs_by_env: dict[str, list[dict]], metric_key: str) -> float:
    """Within-env z-score then concatenate, paper Section 4 protocol."""
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    for env, logs in seed_logs_by_env.items():
        for sl in logs:
            m = np.asarray(sl["log"][metric_key], dtype=float)
            dj = np.asarray(sl["log"]["delta_J_k"], dtype=float)
            mask = np.isfinite(m) & np.isfinite(dj)
            m = m[mask]; dj = dj[mask]
            if m.size > 1 and np.std(m) > 0 and np.std(dj) > 0:
                xs.append((m - np.mean(m)) / np.std(m))
                ys.append((dj - np.mean(dj)) / np.std(dj))
    if not xs:
        return float("nan")
    return pearson(np.concatenate(xs), np.concatenate(ys)).r


def main() -> None:
    print("=" * 110)
    print("MAIN GRID (8 envs × 20 seeds)  →  paper Table 1")
    print("=" * 110)
    header = (
        f"{'env':<16} {'cg AUROC':>10} {'ε_u AUROC':>10} {'cg AUPRC':>10} "
        f"{'ε_u AUPRC':>10} {'cg wins':>8} {'Wilcoxon p':>11} {'base':>6}  "
        f"{'paper cg':>8} {'paper εu':>9} {'Δcg':>6} {'Δεu':>6}"
    )
    print(header)
    print("-" * len(header))

    seed_logs_by_env: dict[str, list[dict]] = {}
    rows: list[list] = []
    for env in ENV_ORDER:
        seed_logs = load_seed_logs(env)
        if not seed_logs:
            print(f"{env:<16} (no data)")
            continue
        seed_logs_by_env[env] = seed_logs

        cg = per_seed_auroc(seed_logs, "cert_gap_k")
        eps = per_seed_auroc(seed_logs, "eps_u", sign=-1)  # high ε_u → predicts harm
        cg_pr = per_seed_auprc(seed_logs, "cert_gap_k")
        eps_pr = per_seed_auprc(seed_logs, "eps_u", sign=-1)
        wins = int(np.nansum(cg > eps))
        n_pairs = int(np.nansum(np.isfinite(cg) & np.isfinite(eps)))
        _, p = paired_wilcoxon_one_sided_greater(cg, eps)
        base = env_base_rate(seed_logs)

        paper_cg, paper_eps, paper_base = PAPER_TABLE1[env]
        med_cg = float(np.nanmedian(cg))
        med_eps = float(np.nanmedian(eps))
        d_cg = med_cg - paper_cg
        d_eps = med_eps - paper_eps

        print(
            f"{env:<16} {med_cg:>10.3f} {med_eps:>10.3f} "
            f"{float(np.nanmedian(cg_pr)):>10.3f} {float(np.nanmedian(eps_pr)):>10.3f} "
            f"{wins:>4}/{n_pairs:<3} {p:>11.2e} {base:>6.2f}  "
            f"{paper_cg:>8.2f} {paper_eps:>9.2f} {d_cg:>+6.2f} {d_eps:>+6.2f}"
        )
        rows.append([env, len(seed_logs), med_cg, med_eps, float(np.nanmedian(cg_pr)),
                     float(np.nanmedian(eps_pr)), wins, n_pairs, p, base,
                     paper_cg, paper_eps, paper_base])

    # Pooled Pearson r (paper Figure 2).
    r_cg = pooled_pearson(seed_logs_by_env, "cert_gap_k")
    r_eps = pooled_pearson(seed_logs_by_env, "eps_u")
    print()
    print(f"Pooled Pearson r (cert-gap vs ΔJ, z-scored within env): r = {r_cg:+.3f}   (paper: +0.560)")
    print(f"Pooled Pearson r (ε_u       vs ΔJ, z-scored within env): r = {r_eps:+.3f}   (paper: +0.085)")

    # Pooled Wilcoxon over all 8 × 20 = 160 seeds.
    cg_all = np.concatenate([per_seed_auroc(seed_logs_by_env[e], "cert_gap_k") for e in seed_logs_by_env])
    eps_all = np.concatenate([per_seed_auroc(seed_logs_by_env[e], "eps_u", sign=-1) for e in seed_logs_by_env])
    pooled_wins = int(np.nansum(cg_all > eps_all))
    pooled_n = int(np.nansum(np.isfinite(cg_all) & np.isfinite(eps_all)))
    _, pooled_p = paired_wilcoxon_one_sided_greater(cg_all, eps_all)
    print(f"Pooled per-seed dominance: cg > ε_u in {pooled_wins}/{pooled_n} seeds, "
          f"Wilcoxon p={pooled_p:.2e}   (paper: 83/85, p<10⁻¹⁵)")
    print(f"Pooled cg AUROC median: {float(np.nanmedian(cg_all)):.3f}   (paper: 0.83)")
    print(f"Pooled ε_u AUROC median: {float(np.nanmedian(eps_all)):.3f}   (paper: 0.45)")

    # Save CSV
    out_csv = Path("results/preliminary_table1.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["env", "n_seeds", "cg_AUROC_med", "eps_u_AUROC_med",
                    "cg_AUPRC_med", "eps_u_AUPRC_med",
                    "cg_wins", "n_pairs", "wilcoxon_p", "base_rate",
                    "paper_cg", "paper_eps_u", "paper_base"])
        w.writerows(rows)
    print(f"\nwrote {out_csv}")

    # ----------------------- tabular identity ----------------------------------
    print()
    print("=" * 110)
    print("TABULAR IDENTITY (Appendix A)")
    print("=" * 110)
    all_residuals: list[float] = []
    by_family: dict[str, list[float]] = {}
    for fam_dir in sorted(Path("results/tabular").iterdir()):
        if not fam_dir.is_dir():
            continue
        fam_residuals = []
        for csv_path in sorted(fam_dir.glob("*.csv")):
            with open(csv_path) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    fam_residuals.append(float(row["identity_error"]))
        if fam_residuals:
            arr = np.asarray(fam_residuals, dtype=float)
            print(f"  {fam_dir.name}: n={arr.size}, median={np.median(arr):.2e}, max={arr.max():.2e}")
            by_family[fam_dir.name] = fam_residuals
            all_residuals.extend(fam_residuals)
    if all_residuals:
        arr = np.asarray(all_residuals, dtype=float)
        print(f"\n  ALL families: n={arr.size}, median={np.median(arr):.2e}, "
              f"mean={arr.mean():.2e}, max={arr.max():.2e}")
        print(f"  paper claim: median ~5e-15, max ~4e-14")
        print(f"  → {'PASS' if arr.max() < 1e-13 else 'FAIL'} (max should be < 1e-13)")
        # save for fig5
        import json
        out_json = Path("results/tabular/identity_residuals.json")
        out_json.write_text(json.dumps(by_family))
        print(f"\n  wrote {out_json}")

    # ----------------------- factorials (partial) -------------------------------
    print()
    print("=" * 110)
    print("FACTORIAL CELLS (whatever has data so far)")
    print("=" * 110)
    for folder, name in [
        (Path("results/factorial_lunarlander"), "LunarLander 4×4"),
        (Path("results/factorial_halfcheetah"), "HalfCheetah 3×2"),
    ]:
        if not folder.exists():
            print(f"  {name}: no data")
            continue
        print(f"\n  --- {name} ---")
        for cell_dir in sorted(folder.iterdir()):
            if not cell_dir.is_dir():
                continue
            cg_chunks: list[np.ndarray] = []
            harm_chunks: list[np.ndarray] = []
            eps_chunks: list[np.ndarray] = []
            n_seeds = 0
            for pkl in sorted(cell_dir.glob("seed_*.pkl")):
                with open(pkl, "rb") as f:
                    log = pickle.load(f)["log"]
                cg_chunks.append(np.asarray(log["cert_gap_k"], dtype=float))
                eps_chunks.append(-np.asarray(log["eps_u"], dtype=float))
                harm_chunks.append(np.asarray(log["harmful_k"], dtype=float))
                n_seeds += 1
            if not cg_chunks:
                continue
            cg = np.concatenate(cg_chunks)
            eps = np.concatenate(eps_chunks)
            harm = np.concatenate(harm_chunks)
            print(f"    {cell_dir.name}: seeds={n_seeds}  "
                  f"cg AUROC={auroc(cg, harm):.3f}  ε_u AUROC={auroc(eps, harm):.3f}")


if __name__ == "__main__":
    main()
