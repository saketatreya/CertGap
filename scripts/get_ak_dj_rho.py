import pickle
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr

all_A = []
all_dj = []
for p in Path("results/main/Humanoid-v5").glob("seed_*.pkl"):
    with open(p, "rb") as f:
        log = pickle.load(f)["log"]
        A = log["A_k"]
        dj = log["delta_J_k"]
        idx = np.isfinite(A) & np.isfinite(dj)
        all_A.extend(np.asarray(A)[idx])
        all_dj.extend(np.asarray(dj)[idx])

rho, _ = spearmanr(all_A, all_dj)
print(f"Ak Paradox rho: {rho:.3f}")
