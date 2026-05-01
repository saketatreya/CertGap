import pickle
from pathlib import Path
import numpy as np

def load_final_returns(folder):
    rets = []
    for p in Path(folder).glob("seed_*.pkl"):
        with open(p, "rb") as f:
            log = pickle.load(f)["log"]
            rets.append(log["J_k"][-1])
    return np.mean(rets), np.std(rets)

v_m, v_s = load_final_returns("results/intervention_check/vanilla")
g_m, g_s = load_final_returns("results/intervention_check/gated")
print(f"Vanilla: {v_m:.1f} +/- {v_s:.1f}")
print(f"Gated:   {g_m:.1f} +/- {g_s:.1f}")
