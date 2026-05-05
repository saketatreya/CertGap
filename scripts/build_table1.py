"""Build the paper Table 1 — cross-algorithm AUROC for Bellman residual vs.
start-state critic bias as per-update harm predictors.

Writes:
  - figures/out/table1.csv
  - figures/out/table1.tex

Rows: one per (algorithm, environment) cell.
Cols: n seeds, AUROC of -ε_u, AUROC of -Δ̂_k, paired Wilcoxon p (Δ̂ > -ε_u),
      base rate.
Plus per-algorithm pooled summary and an overall pooled row.
"""

from __future__ import annotations

import csv
import pickle
from pathlib import Path

import numpy as np

from certgap.analysis.correlations import paired_wilcoxon_one_sided_greater
from certgap.analysis.harm_prediction import auroc


# (algorithm display label, results subfolder, env list, header label)
ALGOS = [
    ("PPO",     "main", [
        "LunarLander-v3", "CartPole-v1", "Acrobot-v1",
        "Hopper-v5", "HalfCheetah-v5", "Walker2d-v5", "Ant-v5", "Humanoid-v5",
    ]),
    ("PPO-MSE", "ppo_mse", [
        "LunarLander-v3", "CartPole-v1", "Acrobot-v1",
        "Hopper-v5", "HalfCheetah-v5", "Walker2d-v5", "Ant-v5", "Humanoid-v5",
    ]),
    ("PPO-NU",  "ppo_nu_weighted", ["Hopper-v5", "HalfCheetah-v5", "Humanoid-v5"]),
    ("SAC",     "sac",  ["Hopper-v5", "HalfCheetah-v5", "Walker2d-v5", "Ant-v5", "Humanoid-v5"]),
    ("TRPO",    "trpo", ["LunarLander-v3", "Hopper-v5"]),
]


def load_seeds(folder: Path) -> list[dict]:
    out = []
    for p in sorted(Path(folder).glob("seed_*.pkl")):
        with open(p, "rb") as f:
            d = pickle.load(f)
        if d["metadata"]["n_updates"] < 10:
            continue
        out.append(d)
    return out


def per_seed_auroc(seeds: list[dict], key: str, sign: int) -> np.ndarray:
    out = []
    for d in seeds:
        log = d["log"]
        if key not in log:
            out.append(np.nan); continue
        m = sign * np.asarray(log[key], dtype=float)
        h = np.asarray(log["harmful_k"], dtype=float)
        out.append(auroc(m, h))
    return np.asarray(out, dtype=float)


def base_rate(seeds: list[dict]) -> float:
    rates = []
    for d in seeds:
        h = np.asarray(d["log"]["harmful_k"], dtype=float)
        h = h[np.isfinite(h)]
        if h.size:
            rates.append(float(h.mean()))
    return float(np.mean(rates)) if rates else float("nan")


def main() -> None:
    rows: list[dict] = []
    pooled = {algo: {"eps": [], "dh": []} for algo, _, _ in ALGOS}
    grand = {"eps": [], "dh": []}

    for algo, top, envs in ALGOS:
        for env in envs:
            seeds = load_seeds(Path(f"results/{top}/{env}"))
            if not seeds:
                continue
            eps_arr = per_seed_auroc(seeds, "eps_u", -1)
            dh_arr = per_seed_auroc(seeds, "delta_hat_k", -1)
            pooled[algo]["eps"].extend(eps_arr.tolist())
            pooled[algo]["dh"].extend(dh_arr.tolist())
            if algo != "PPO-NU":
                grand["eps"].extend(eps_arr.tolist())
                grand["dh"].extend(dh_arr.tolist())
            mask = np.isfinite(eps_arr) & np.isfinite(dh_arr)
            wins = int(np.sum(dh_arr[mask] > eps_arr[mask]))
            n_pairs = int(mask.sum())
            _, p = paired_wilcoxon_one_sided_greater(dh_arr[mask], eps_arr[mask])
            rows.append({
                "algo": algo,
                "env": env,
                "n_seeds": len(seeds),
                "auroc_eps_u_neg": float(np.nanmedian(eps_arr)),
                "auroc_dhat_neg": float(np.nanmedian(dh_arr)),
                "wins": f"{wins}/{n_pairs}",
                "wilcoxon_p": p,
                "base_rate": base_rate(seeds),
            })

    # Per-algo pooled summary rows
    summary_rows = []
    for algo, _, _ in ALGOS:
        eps_arr = np.asarray(pooled[algo]["eps"], dtype=float)
        dh_arr = np.asarray(pooled[algo]["dh"], dtype=float)
        if eps_arr.size == 0:
            continue
        mask = np.isfinite(eps_arr) & np.isfinite(dh_arr)
        wins = int(np.sum(dh_arr[mask] > eps_arr[mask]))
        n_pairs = int(mask.sum())
        _, p = paired_wilcoxon_one_sided_greater(dh_arr[mask], eps_arr[mask])
        summary_rows.append({
            "algo": algo,
            "env": f"-- pooled ({eps_arr.size} seeds) --",
            "n_seeds": eps_arr.size,
            "auroc_eps_u_neg": float(np.nanmedian(eps_arr)),
            "auroc_dhat_neg": float(np.nanmedian(dh_arr)),
            "wins": f"{wins}/{n_pairs}",
            "wilcoxon_p": p,
            "base_rate": float("nan"),
        })

    # Grand pooled row
    eps_all = np.asarray(grand["eps"], dtype=float)
    dh_all = np.asarray(grand["dh"], dtype=float)
    mask = np.isfinite(eps_all) & np.isfinite(dh_all)
    wins = int(np.sum(dh_all[mask] > eps_all[mask]))
    n_pairs = int(mask.sum())
    _, p_all = paired_wilcoxon_one_sided_greater(dh_all[mask], eps_all[mask])
    grand_row = {
        "algo": "ALL",
        "env": f"-- pooled ({eps_all.size} seeds, 4 algos) --",
        "n_seeds": eps_all.size,
        "auroc_eps_u_neg": float(np.nanmedian(eps_all)),
        "auroc_dhat_neg": float(np.nanmedian(dh_all)),
        "wins": f"{wins}/{n_pairs}",
        "wilcoxon_p": p_all,
        "base_rate": float("nan"),
    }

    out_dir = Path("figures/out")
    out_dir.mkdir(parents=True, exist_ok=True)

    # CSV
    csv_rows = rows + summary_rows + [grand_row]
    csv_path = out_dir / "table1.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
        w.writeheader()
        w.writerows(csv_rows)
    print(f"wrote {csv_path}")

    # LaTeX
    lines: list[str] = [
        r"% Generated by scripts/build_table1.py — do not edit by hand.",
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{\textbf{Per-environment median harm-prediction AUROC, by algorithm.} "
        r"For each (algorithm, environment) cell we compute the per-seed AUROC of $-\widehat\varepsilon_u$ and "
        r"$-\widehat{\hat\Delta}_k$ for predicting $\Delta J_k<0$ on that seed; the table reports seed medians. "
        r"`$\hat\Delta$ wins' is the per-seed paired count favoring $-\hat\Delta_k$; "
        r"$p$ is the one-sided paired Wilcoxon ($-\hat\Delta_k > -\varepsilon_u$). "
        r"`base' is the harmful-update fraction (chance AUROC is 0.5; chance AUPRC equals the base rate). "
        r"Cells with $n<5$ seeds report `---' for the Wilcoxon $p$.}",
        r"\label{tab:per-env-auroc}",
        r"\begin{tabular}{llrrrrrr}",
        r"\toprule",
        r"Algorithm & Environment & $n$ & $-\varepsilon_u$ & $-\hat\Delta_k$ & "
        r"$\hat\Delta$ wins & $p$ & base \\",
        r"\midrule",
    ]
    last_algo = None
    for r in rows:
        if r["algo"] != last_algo:
            if last_algo is not None:
                lines.append(r"\midrule")
            last_algo = r["algo"]
        env_short = r["env"].split("-")[0]
        p = r["wilcoxon_p"]
        if not np.isfinite(p):
            p_str = "---"
        elif p < 1e-4:
            p_str = f"{p:.0e}"
        else:
            p_str = f"{p:.3f}"
        lines.append(
            f"{r['algo']:<8} & {env_short:<14} & {r['n_seeds']:>2} & "
            f"{r['auroc_eps_u_neg']:.2f} & "
            f"\\textbf{{{r['auroc_dhat_neg']:.2f}}} & "
            f"{r['wins']} & {p_str} & {r['base_rate']:.2f} \\\\"
        )
    lines.append(r"\midrule")
    for r in summary_rows:
        algo = r["algo"]
        p = r["wilcoxon_p"]
        if not np.isfinite(p):
            p_str = "---"
        elif p < 1e-4:
            p_str = f"{p:.0e}"
        else:
            p_str = f"{p:.3f}"
        lines.append(
            f"\\textit{{pooled}} & \\textit{{{algo}}} & {r['n_seeds']:>3} & "
            f"{r['auroc_eps_u_neg']:.2f} & "
            f"\\textbf{{{r['auroc_dhat_neg']:.2f}}} & "
            f"{r['wins']} & {p_str} & --- \\\\"
        )
    lines.append(r"\midrule")
    p = grand_row["wilcoxon_p"]
    p_str = f"{p:.0e}" if np.isfinite(p) else "---"
    lines.append(
        f"\\textbf{{ALL}} & \\textbf{{(4 algos, 23 cells)}} & "
        f"\\textbf{{{grand_row['n_seeds']}}} & "
        f"\\textbf{{{grand_row['auroc_eps_u_neg']:.2f}}} & "
        f"\\textbf{{{grand_row['auroc_dhat_neg']:.2f}}} & "
        f"\\textbf{{{grand_row['wins']}}} & \\textbf{{{p_str}}} & --- \\\\"
    )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]

    tex_path = out_dir / "table1.tex"
    tex_path.write_text("\n".join(lines))
    print(f"wrote {tex_path}")

    # Also print the table to stdout for quick visual check
    print()
    print(f"{'algo':<8} {'env':<14} {'n':>3} {'-eps_u':>7} {'-Delta':>7}  {'wins':>7}  {'p':>10}  {'base':>5}")
    print("-" * 70)
    for r in rows:
        env_short = r["env"].split("-")[0]
        p = r["wilcoxon_p"]
        p_str = f"{p:.1e}" if np.isfinite(p) else "---"
        print(f"{r['algo']:<8} {env_short:<14} {r['n_seeds']:>3} "
              f"{r['auroc_eps_u_neg']:>7.3f} {r['auroc_dhat_neg']:>7.3f}  "
              f"{r['wins']:>7}  {p_str:>10}  {r['base_rate']:>5.2f}")
    print("-" * 70)
    for r in summary_rows + [grand_row]:
        p = r["wilcoxon_p"]
        p_str = f"{p:.1e}" if np.isfinite(p) else "---"
        print(f"{r['algo']:<8} {r['env']:<25} {r['n_seeds']:>3} "
              f"{r['auroc_eps_u_neg']:>7.3f} {r['auroc_dhat_neg']:>7.3f}  "
              f"{r['wins']:>7}  {p_str:>10}")


if __name__ == "__main__":
    main()
