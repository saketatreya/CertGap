#!/usr/bin/env bash
# Hopper value-epochs x clip-epsilon factorial plus value-lr sweep.
set -euo pipefail
cd "$(dirname "$0")/.."

WORKERS=${WORKERS:-$(( $(sysctl -n hw.ncpu 2>/dev/null || nproc) / 2 ))}
SEEDS=${SEEDS:-5}
python -m certgap.runners.sweep \
  --grid-name ppo_factorial_hopper --seeds "$SEEDS" --workers "$WORKERS" \
  --status results/run_status_factorial_hopper.json
