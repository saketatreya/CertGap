#!/usr/bin/env bash
# Bellman-residual scalarization check on Humanoid-v5.
set -euo pipefail
cd "$(dirname "$0")/.."

WORKERS=${WORKERS:-$(( $(sysctl -n hw.ncpu 2>/dev/null || nproc) / 2 ))}
SEEDS=${SEEDS:-6}
python -m certgap.runners.sweep \
  --grid-name ppo_eps_u_variants --seeds "$SEEDS" --workers "$WORKERS" \
  --status results/run_status_eps_u_variants.json
