#!/usr/bin/env bash
# Humanoid-v5 train-vs-held-out residual estimator agreement.
set -euo pipefail
cd "$(dirname "$0")/.."

WORKERS=${WORKERS:-$(( $(sysctl -n hw.ncpu 2>/dev/null || nproc) / 2 ))}
SEEDS=${SEEDS:-20}
python -m certgap.runners.sweep \
  --grid-name ppo_heldout_humanoid --seeds "$SEEDS" --workers "$WORKERS" \
  --status results/run_status_heldout_humanoid.json
