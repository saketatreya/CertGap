"""Single training run.

    python -m certgap.runners.run_one \\
        --algo ppo \\
        --env Humanoid-v5 --seed 0 --total-timesteps 500000 \\
        --output results/main/Humanoid-v5/seed_0.pkl \\
        --log-heldout

Writes a pickle containing {"metadata": ..., "log": {key: np.ndarray}}.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from certgap.common.config import PPOConfig
from certgap.common.utils import save_pickle, save_json
from certgap.ppo import train_ppo
from certgap.sac import SACConfig, train_sac
from certgap.trpo import TRPOConfig, train_trpo


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Single training seed.")
    p.add_argument("--algo", choices=("ppo", "sac", "trpo"), default="ppo")
    p.add_argument("--env", required=True, help="gymnasium env id, e.g. Hopper-v5")
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--output", required=True, help="path for the result pickle")
    p.add_argument("--total-timesteps", type=int, default=PPOConfig.total_timesteps)
    p.add_argument("--n-steps", type=int, default=PPOConfig.n_steps)
    p.add_argument("--minibatch-size", type=int, default=PPOConfig.minibatch_size)
    p.add_argument("--policy-epochs", type=int, default=PPOConfig.policy_epochs)
    p.add_argument("--value-epochs", type=int, default=PPOConfig.value_epochs)
    p.add_argument("--clip-eps", type=float, default=PPOConfig.clip_eps)
    p.add_argument("--policy-lr", type=float, default=PPOConfig.policy_lr)
    p.add_argument("--value-lr", type=float, default=PPOConfig.value_lr)
    p.add_argument("--gamma", type=float, default=PPOConfig.gamma)
    p.add_argument("--gae-lambda", type=float, default=PPOConfig.gae_lambda)
    p.add_argument("--eps-u-kind", choices=("mse", "sup", "p95"), default=PPOConfig.eps_u_kind)
    p.add_argument(
        "--log-heldout",
        action="store_true",
        help="Collect a 500-transition held-out batch each update (figA).",
    )
    p.add_argument(
        "--heldout-batch-size",
        type=int,
        default=500,
        help="Held-out batch size when --log-heldout.",
    )
    p.add_argument(
        "--log-eps-u-variants",
        action="store_true",
        help="Log all three ε_u definitions (mse, sup, p95) per update (figB).",
    )
    p.add_argument(
        "--factorial-value-epochs", type=int, default=None,
        help="Override value_epochs for hyperparameter factorial cells.",
    )
    p.add_argument(
        "--factorial-clip-eps", type=float, default=None,
        help="Override clip_eps for hyperparameter factorial cells.",
    )
    p.add_argument(
        "--factorial-value-lr", type=float, default=None,
        help="Override value_lr for hyperparameter factorial cells.",
    )
    p.add_argument("--gating-threshold", type=float, default=None, help="Start-state bias threshold for update rejection.")
    p.add_argument("--nu-loss-weight", type=float, default=None,
                   help="Weight for the ν-weighted critic loss term (0 = vanilla PPO).")
    p.add_argument("--quiet", action="store_true", help="Suppress per-update stdout logging.")
    p.add_argument("--skip-if-exists", action="store_true",
                   help="If output already exists, exit successfully without rerunning.")
    p.add_argument("--checkpoint-interval", type=int, default=SACConfig.checkpoint_interval)
    p.add_argument("--eval-episodes", type=int, default=SACConfig.eval_episodes)
    p.add_argument("--learning-starts", type=int, default=SACConfig.learning_starts)
    p.add_argument("--train-freq", type=int, default=SACConfig.train_freq)
    p.add_argument("--batch-size", type=int, default=SACConfig.batch_size)
    p.add_argument("--replay-size", type=int, default=SACConfig.replay_size)
    p.add_argument("--alpha", type=float, default=SACConfig.alpha)
    p.add_argument("--target-kl", type=float, default=TRPOConfig.target_kl)
    p.add_argument("--cg-iters", type=int, default=TRPOConfig.cg_iters)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output = Path(args.output)
    if args.skip_if_exists and output.exists():
        print(f"[skip] {output} exists.")
        return 0

    if args.algo == "ppo":
        cfg = PPOConfig(
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            total_timesteps=args.total_timesteps,
            n_steps=args.n_steps,
            minibatch_size=args.minibatch_size,
            policy_epochs=args.policy_epochs,
            value_epochs=args.value_epochs,
            clip_eps=args.clip_eps,
            policy_lr=args.policy_lr,
            value_lr=args.value_lr,
            eps_u_kind=args.eps_u_kind,
            heldout_batch_size=args.heldout_batch_size if args.log_heldout else 0,
            gating_threshold=args.gating_threshold,
            nu_loss_weight=args.nu_loss_weight if args.nu_loss_weight is not None else 0.0,
        )

        result = train_ppo(
            env_id=args.env,
            seed=args.seed,
            cfg=cfg,
            log_heldout=args.log_heldout,
            log_eps_u_variants=args.log_eps_u_variants,
            factorial_value_epochs=args.factorial_value_epochs,
            factorial_clip_eps=args.factorial_clip_eps,
            factorial_value_lr=args.factorial_value_lr,
            verbose=not args.quiet,
        )
    elif args.algo == "sac":
        cfg = SACConfig(
            gamma=args.gamma,
            total_timesteps=args.total_timesteps,
            batch_size=args.batch_size,
            replay_size=args.replay_size,
            learning_starts=args.learning_starts,
            train_freq=args.train_freq,
            policy_lr=args.policy_lr,
            q_lr=args.value_lr,
            alpha=args.alpha,
            checkpoint_interval=args.checkpoint_interval,
            eval_episodes=args.eval_episodes,
        )
        result = train_sac(
            env_id=args.env,
            seed=args.seed,
            cfg=cfg,
            verbose=not args.quiet,
            heartbeat_path=output.with_suffix(".heartbeat"),
        )
    else:
        cfg = TRPOConfig(
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            total_timesteps=args.total_timesteps,
            n_steps=args.n_steps,
            minibatch_size=args.minibatch_size,
            value_epochs=args.value_epochs,
            value_lr=args.value_lr,
            eps_u_kind=args.eps_u_kind,
            target_kl=args.target_kl,
            cg_iters=args.cg_iters,
        )
        result = train_trpo(
            env_id=args.env,
            seed=args.seed,
            cfg=cfg,
            verbose=not args.quiet,
        )

    save_pickle(result, output)
    log = result["log"]
    cert_gap = log.get("cert_gap_k")
    median_cert_gap = (
        float(np.nanmedian(cert_gap))
        if cert_gap is not None and np.isfinite(cert_gap).any()
        else float("nan")
    )
    summary = {
        "algo": result["metadata"].get("algo", args.algo),
        "env_id": result["metadata"]["env_id"],
        "resolved_env_id": result["metadata"]["resolved_env_id"],
        "seed": result["metadata"]["seed"],
        "n_updates": result["metadata"]["n_updates"],
        "runtime_seconds": result["metadata"]["runtime_seconds"],
        "final_J": float(result["log"]["J_k"][-1]),
        "median_cert_gap": median_cert_gap,
        "median_eps_u": float(np.nanmedian(result["log"]["eps_u"])),
    }
    save_json(summary, output.with_suffix(".json"))
    print(f"[done] {output}  ({summary['n_updates']} updates, {summary['runtime_seconds']:.0f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
