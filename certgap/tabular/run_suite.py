"""Suite runner for tabular experiments.

Collects results for RandomMDP across multiple seeds and hyperparameter families.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import tqdm

from certgap.tabular.train import train_bridge

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", type=Path, default=Path("results/tabular"))
    p.add_argument("--seeds", type=int, default=8)
    p.add_argument("--iterations", type=int, default=100)
    p.add_argument("--skip-if-exists", action="store_true")
    args = p.parse_args()

    families = [
        {"S": 64, "A": 4, "gamma": 0.95, "name": "family_1"},
        {"S": 128, "A": 8, "gamma": 0.90, "name": "family_2"},
        {"S": 32, "A": 2, "gamma": 0.99, "name": "family_3"},
    ]

    total_runs = len(families) * args.seeds
    pbar = tqdm.tqdm(total=total_runs, desc="Tabular Suite")

    for fam in families:
        fam_dir = args.output_dir / fam["name"]
        for seed in range(args.seeds):
            train_bridge(
                mdp_seed=seed,
                n_states=fam["S"],
                n_actions=fam["A"],
                n_iterations=args.iterations,
                gamma=fam["gamma"],
                output_dir=str(fam_dir),
                skip_if_exists=args.skip_if_exists,
            )
            pbar.update(1)
    pbar.close()

if __name__ == "__main__":
    main()
