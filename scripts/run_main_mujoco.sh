#!/usr/bin/env bash
# Primary PPO grid entrypoint.
set -euo pipefail
cd "$(dirname "$0")/.."

WORKERS=${WORKERS:-$(( $(sysctl -n hw.ncpu 2>/dev/null || nproc) / 2 ))}
SEEDS=${SEEDS:-20}
echo "[run_main_mujoco] WORKERS=$WORKERS SEEDS=$SEEDS"

python -m certgap.runners.sweep \
  --grid-name ppo_main --seeds "$SEEDS" --workers "$WORKERS" \
  --status results/run_status_main_mujoco.json
