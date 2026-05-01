#!/usr/bin/env bash
# Figure 5 / Appendix A: tabular RandomMDP × NPG, identity residual.
# Three (|S|, |A|, γ) families × 8 seeds × 100 NPG iterations.
set -euo pipefail
cd "$(dirname "$0")/.."

python -m certgap.tabular.run_suite \
  --output-dir results/tabular \
  --seeds 8 --iterations 100
