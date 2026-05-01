#!/usr/bin/env bash
# LunarLander value-epochs x clip-epsilon factorial.
set -euo pipefail
cd "$(dirname "$0")/.."

WORKERS=${WORKERS:-$(( $(sysctl -n hw.ncpu 2>/dev/null || nproc) / 2 ))}
SEEDS=${SEEDS:-5}
python -m certgap.runners.sweep \
  --grid-name ppo_factorial_lunarlander --seeds "$SEEDS" --workers "$WORKERS" \
  --status results/run_status_factorial_lunarlander.json
