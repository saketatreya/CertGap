import pickle
from pathlib import Path
import numpy as np

total_updates = 0
total_rejected = 0
for p in Path("results/intervention_check/gated").glob("seed_*.pkl"):
    with open(p, "rb") as f:
        log = pickle.load(f)["log"]
        rej = np.asarray(log["rejected"])
        total_updates += len(rej)
        total_rejected += np.sum(rej > 0.5)

print(f"Rejection Rate: {100 * total_rejected / total_updates:.1f}%")
