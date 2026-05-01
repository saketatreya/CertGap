"""Action A: Fully Independent Held-out AUROC.

Computes the AUROC of delta_hat_heldout_k against delta_J_heldout_k.
Both are computed on a separate evaluation rollout, ensuring 0% batch leakage.
"""

from __future__ import annotations

import numpy as np
from certgap.analysis.harm_prediction import auroc
from figures._common import load_pickle

def main() -> None:
    path = "results/heldout_humanoid_v2.pkl"
    data = load_pickle(path)
    log = data["log"]
    
    # delta_hat_heldout_k was computed on the heldout rollout using the residual estimator.
    # J_heldout_k is the mean episode return on that same heldout rollout.
    m = log["delta_hat_heldout_k"]
    j = log["J_heldout_k"]
    
    # Compute delta_J_heldout
    dj = j[1:] - j[:-1]
    harm = (dj < 0).astype(float)
    
    # Metric at step k predicts harm at step k (concurrent)
    # Both m and dj are "concurrent" here because m was computed AFTER the critic update
    # and dj is the change resulting from the policy update.
    # Wait, in ppo.py, update order is:
    # 1. collect B_k (gives J_k)
    # 2. update policy (gives pi_{k+1})
    # 3. update critic (gives V_{k+1})
    # 4. compute delta_hat on B_heldout using V_{k+1}.
    # 5. next loop: collect B_{k+1} (gives J_{k+1}).
    
    # So m[k] is the diagnostic for the update that just happened.
    # The outcome of that update is J[k+1] - J[k].
    
    m_valid = -m[:-1] # negative bias predicts improvement
    
    auc = auroc(m_valid, harm)
    print(f"Independent Held-out AUROC (Humanoid-v5, n=20): {auc:.3f}")

if __name__ == "__main__":
    main()
