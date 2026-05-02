"""Multiprocess pool driver for experiment grids.
"""
from __future__ import annotations
import argparse
import itertools
import json
import multiprocessing as mp
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any
import tqdm
from certgap.common.utils import ensure_dir, save_json
from certgap.runners.run_one import main as run_one_main

def _ppo_main_envs() -> list[tuple[str, int]]:
    return [
        ("LunarLander-v3", 300000), ("CartPole-v1", 200000), ("Acrobot-v1", 300000),
        ("Hopper-v5", 300000), ("HalfCheetah-v5", 300000), ("Walker2d-v5", 300000),
        ("Ant-v5", 300000), ("Humanoid-v5", 500000)
    ]

def grid_ppo_main(seeds=20):
    runs = []
    for env, ts in _ppo_main_envs():
        for s in range(seeds):
            runs.append({"args": ["--algo", "ppo", "--env", env, "--seed", str(s), "--total-timesteps", str(ts), "--output", f"results/main/{env}/seed_{s}.pkl", "--skip-if-exists", "--quiet"]})
    return runs

def grid_ppo_mse(seeds=5):
    runs = []
    for env, ts in _ppo_main_envs():
        for s in range(seeds):
            runs.append({"args": ["--algo", "ppo", "--env", env, "--seed", str(s), "--total-timesteps", str(ts), "--eps-u-kind", "mse", "--log-eps-u-variants", "--output", f"results/ppo_mse/{env}/seed_{s}.pkl", "--skip-if-exists", "--quiet"]})
    return runs

def grid_sac_main(seeds=10):
    runs = []
    for env in ["Hopper-v5", "HalfCheetah-v5", "Walker2d-v5"]:
        for s in range(seeds):
            runs.append({"args": ["--algo", "sac", "--env", env, "--seed", str(s), "--total-timesteps", "200000", "--output", f"results/sac/{env}/seed_{s}.pkl", "--skip-if-exists", "--quiet"]})
    return runs

def grid_trpo_pilot(seeds=10):
    runs = []
    for env in ["LunarLander-v3", "Hopper-v5"]:
        for s in range(seeds):
            runs.append({"args": ["--algo", "trpo", "--env", env, "--seed", str(s), "--total-timesteps", "150000", "--output", f"results/trpo/{env}/seed_{s}.pkl", "--skip-if-exists", "--quiet"]})
    return runs

def grid_ppo_intervention_check(seeds=5):
    runs = []
    for s in range(seeds):
        runs.append({"args": ["--algo", "ppo", "--env", "Humanoid-v5", "--seed", str(s), "--total-timesteps", "500000", "--output", f"results/intervention_check/vanilla/seed_{s}.pkl", "--skip-if-exists", "--quiet"]})
    for s in range(seeds):
        runs.append({"args": ["--algo", "ppo", "--env", "Humanoid-v5", "--seed", str(s), "--total-timesteps", "500000", "--gating-threshold", "161", "--output", f"results/intervention_check/gated/seed_{s}.pkl", "--skip-if-exists", "--quiet"]})
    return runs

def grid_recompute_baselines():
    runs = []
    for env, ts in [("Humanoid-v5", 100000), ("Hopper-v5", 100000), ("HalfCheetah-v5", 100000)]:
        runs.append({"args": ["--algo", "ppo", "--env", env, "--seed", "0", "--total-timesteps", str(ts), "--output", f"results/main/{env}/seed_0.pkl", "--quiet"]})
    return runs


def _factorial_grid(env: str, total_steps: int, ve_values: list[int], ce_values: list[float],
                    vlr_values: list[float] | None, out_dir: str, seeds: int = 5):
    runs = []
    for ve in ve_values:
        for ce in ce_values:
            cell = f"ve{ve}_ce{ce}"
            for s in range(seeds):
                runs.append({"args": [
                    "--algo", "ppo", "--env", env, "--seed", str(s),
                    "--total-timesteps", str(total_steps),
                    "--factorial-value-epochs", str(ve),
                    "--factorial-clip-eps", str(ce),
                    "--output", f"results/{out_dir}/{cell}/seed_{s}.pkl",
                    "--skip-if-exists", "--quiet",
                ]})
    if vlr_values:
        for vlr in vlr_values:
            cell = f"vlr{vlr:.0e}"
            for s in range(seeds):
                runs.append({"args": [
                    "--algo", "ppo", "--env", env, "--seed", str(s),
                    "--total-timesteps", str(total_steps),
                    "--factorial-value-lr", str(vlr),
                    "--output", f"results/{out_dir}/{cell}/seed_{s}.pkl",
                    "--skip-if-exists", "--quiet",
                ]})
    return runs


def grid_ppo_factorial_lunarlander(seeds=5):
    return _factorial_grid(
        env="LunarLander-v3", total_steps=300000,
        ve_values=[5, 10, 20, 40], ce_values=[0.1, 0.2, 0.3, 0.4],
        vlr_values=None, out_dir="factorial_lunarlander", seeds=seeds,
    )


def grid_ppo_factorial_halfcheetah(seeds=5):
    return _factorial_grid(
        env="HalfCheetah-v5", total_steps=300000,
        ve_values=[5, 10, 20], ce_values=[0.1, 0.2],
        vlr_values=[3e-4, 1e-3, 3e-3], out_dir="factorial_halfcheetah", seeds=seeds,
    )


def grid_ppo_factorial_hopper(seeds=5):
    return _factorial_grid(
        env="Hopper-v5", total_steps=300000,
        ve_values=[5, 10, 20], ce_values=[0.1, 0.2],
        vlr_values=[3e-4, 1e-3, 3e-3], out_dir="factorial_hopper", seeds=seeds,
    )


def grid_ppo_heldout_humanoid(seeds=20):
    runs = []
    for s in range(seeds):
        runs.append({"args": [
            "--algo", "ppo", "--env", "Humanoid-v5", "--seed", str(s),
            "--total-timesteps", "500000",
            "--log-heldout", "--heldout-batch-size", "500",
            "--output", f"results/heldout_humanoid/seed_{s}.pkl",
            "--skip-if-exists", "--quiet",
        ]})
    return runs


def grid_ppo_eps_u_variants(seeds=6):
    runs = []
    for s in range(seeds):
        runs.append({"args": [
            "--algo", "ppo", "--env", "Humanoid-v5", "--seed", str(s),
            "--total-timesteps", "500000",
            "--log-eps-u-variants",
            "--output", f"results/eps_u_variants/Humanoid-v5/seed_{s}.pkl",
            "--skip-if-exists", "--quiet",
        ]})
    return runs


def grid_ppo_nu_pilot(seeds=3):
    """Lambda sweep for the ν-weighted critic loss on Humanoid-v5."""
    runs = []
    for lam in [0.1, 0.5, 1.0, 5.0]:
        for s in range(seeds):
            runs.append({"args": [
                "--algo", "ppo", "--env", "Humanoid-v5", "--seed", str(s),
                "--total-timesteps", "500000",
                "--nu-loss-weight", str(lam),
                "--output", f"results/ppo_nu_pilot/lam{lam}/seed_{s}.pkl",
                "--skip-if-exists", "--quiet",
            ]})
    return runs


def grid_ppo_nu_weighted(seeds=5):
    """ν-weighted critic loss on 3 envs spanning the coupling-sign pattern."""
    runs = []
    for env, ts in [("Humanoid-v5", 500000),
                    ("HalfCheetah-v5", 300000),
                    ("Hopper-v5", 300000)]:
        for s in range(seeds):
            runs.append({"args": [
                "--algo", "ppo", "--env", env, "--seed", str(s),
                "--total-timesteps", str(ts),
                "--nu-loss-weight", "1.0",
                "--output", f"results/ppo_nu_weighted/{env}/seed_{s}.pkl",
                "--skip-if-exists", "--quiet",
            ]})
    return runs


GRID_BUILDERS = {
    "ppo_main": grid_ppo_main,
    "ppo_mse": grid_ppo_mse,
    "sac_main": grid_sac_main,
    "trpo_pilot": grid_trpo_pilot,
    "ppo_intervention_check": grid_ppo_intervention_check,
    "ppo_factorial_lunarlander": grid_ppo_factorial_lunarlander,
    "ppo_factorial_halfcheetah": grid_ppo_factorial_halfcheetah,
    "ppo_factorial_hopper": grid_ppo_factorial_hopper,
    "ppo_heldout_humanoid": grid_ppo_heldout_humanoid,
    "ppo_eps_u_variants": grid_ppo_eps_u_variants,
    "recompute_baselines": grid_recompute_baselines,
    "ppo_nu_pilot": grid_ppo_nu_pilot,
    "ppo_nu_weighted": grid_ppo_nu_weighted,
}

def _worker(args):
    try:
        run_one_main(args)
        return {"args": args, "rc": 0}
    except Exception as e:
        return {"args": args, "rc": 1, "error": str(e)}

def run_grid(grid, workers, status_path, use_tqdm=True):
    ensure_dir(status_path.parent)
    args_list = [run["args"] for run in grid]
    results = []
    if workers <= 1:
        for args in args_list:
            results.append(_worker(args))
    else:
        with mp.Pool(workers) as pool:
            results = list(tqdm.tqdm(pool.imap(_worker, args_list), total=len(args_list))) if use_tqdm else pool.map(_worker, args_list)
    save_json({"runs": results}, status_path)
    return 0

def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--grid-name", choices=sorted(GRID_BUILDERS.keys()), required=True)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--status", type=Path, default=Path("results/run_status.json"))
    args = p.parse_args(argv)
    builder = GRID_BUILDERS[args.grid_name]
    grid = builder()
    return run_grid(grid, args.workers, args.status)

if __name__ == "__main__":
    sys.exit(main())
