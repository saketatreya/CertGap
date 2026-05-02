#!/usr/bin/env python3
"""Master runner for the full reproduction suite.

Orchestrates tabular, PPO, and SAC experiments with unified progress bars
and restartable skip-if-exists semantics.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Add the project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from certgap.runners.sweep import GRID_BUILDERS, run_grid
from certgap.tabular.run_suite import main as run_tabular_main

def main():
    parser = argparse.ArgumentParser(description="CertGap Master Runner")
    parser.add_argument("--seeds", type=int, default=20, help="Seeds for PPO sweeps (default 20)")
    parser.add_argument("--tabular-seeds", type=int, default=8, help="Seeds for tabular suite (default 8)")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2), 
                        help="Parallel workers for training sweeps")
    parser.add_argument("--skip-if-exists", action="store_true", help="Skip runs whose output already exists")
    parser.add_argument("--skip-ppo", action="store_true", help="Skip PPO sweeps")
    parser.add_argument("--skip-sac", action="store_true", help="Skip SAC sweeps")
    parser.add_argument("--skip-tabular", action="store_true", help="Skip tabular suite")
    parser.add_argument("--include-trpo", action="store_true", help="Run the optional TRPO pilot if available")
    parser.add_argument("--skip-trpo", action="store_true", help="Skip TRPO even when --include-trpo is set")
    args = parser.parse_args()

    print("Starting reproduction suite")
    print(f"   Workers: {args.workers}, PPO seeds: {args.seeds}, tabular seeds: {args.tabular_seeds}")
    print(f"   Skip if exists: {args.skip_if_exists}")
    print("-" * 60)

    # 1. Tabular Suite
    if not args.skip_tabular:
        print("\n[1/4] Tabular identity checks")
        # run_tabular_main expects argv-like list or uses sys.argv
        # We'll call it with specific args
        original_argv = sys.argv
        tabular_argv = [
            "run_suite.py", 
            "--output-dir", "results/tabular", 
            "--seeds", str(args.tabular_seeds), 
            "--iterations", "100"
        ]
        if args.skip_if_exists:
            tabular_argv.append("--skip-if-exists")
        
        sys.argv = tabular_argv
        try:
            run_tabular_main()
        finally:
            sys.argv = original_argv

    # 2. PPO sweeps
    if not args.skip_ppo:
        print("\n[2/4] PPO experiments")
        grids_to_run = [
            ("ppo_main", "PPO primary grid", None),
            ("ppo_mse", "PPO MSE residual control", 5),
            ("ppo_factorial_lunarlander", "PPO LunarLander factorial", 5),
            ("ppo_factorial_halfcheetah", "PPO HalfCheetah factorial", 5),
            ("ppo_factorial_hopper", "PPO Hopper factorial", 5),
            ("ppo_heldout_humanoid", "PPO held-out Humanoid", None),
            ("ppo_eps_u_variants", "PPO residual scalarizations", 6),
            ("recompute_baselines", "PPO baseline metrics recompute", 1),
            ("ppo_intervention_check", "PPO gated intervention study", 5),
            ("ppo_nu_weighted", "PPO ν-weighted critic loss", 5),
        ]

        for grid_name, description, seeds_override in grids_to_run:
            print(f"\n--- {description} ---")
            builder = GRID_BUILDERS[grid_name]
            seeds = seeds_override if seeds_override is not None else args.seeds
            try:
                grid = builder(seeds)
            except TypeError:
                grid = builder()

            status_path = Path(f"results/run_status_{grid_name}.json")
            run_grid(grid, args.workers, status_path, use_tqdm=True)

    # 3. SAC sweeps
    if not args.skip_sac:
        print("\n[3/4] SAC experiments")
        grid_name = "sac_main"
        builder = GRID_BUILDERS[grid_name]
        grid = builder(10)
        status_path = Path(f"results/run_status_{grid_name}.json")
        run_grid(grid, args.workers, status_path, use_tqdm=True)

    # 4. Optional TRPO pilot
    if args.include_trpo and not args.skip_trpo:
        print("\n[4/4] TRPO pilot")
        if "trpo_pilot" not in GRID_BUILDERS:
            raise SystemExit("TRPO pilot grid is not available in this checkout.")
        grid = GRID_BUILDERS["trpo_pilot"](10)
        status_path = Path("results/run_status_trpo_pilot.json")
        run_grid(grid, args.workers, status_path, use_tqdm=True)
    elif args.include_trpo and args.skip_trpo:
        print("\n[4/4] TRPO pilot skipped")
    else:
        print("\n[4/4] TRPO pilot not requested")

    print("\n" + "=" * 60)
    print("Reproduction runs completed.")
    print("   Results are in the 'results/' directory.")
    print("   To generate figures, run: make figures")
    print("=" * 60)

if __name__ == "__main__":
    main()
