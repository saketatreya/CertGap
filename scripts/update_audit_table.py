import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from certgap.analysis.harm_prediction import auroc

def get_auroc(paths, key, sign=-1):
    scores = []
    for p in paths:
        with open(p, "rb") as f:
            log = pickle.load(f)["log"]
            if key not in log: continue
            m = sign * np.asarray(log[key])
            h = np.asarray(log["harmful_k"])
            idx = np.isfinite(m) & np.isfinite(h)
            if idx.any():
                scores.append(auroc(m[idx], h[idx]))
    return np.median(scores) if scores else np.nan

def main():
    envs = ["Humanoid-v5", "Hopper-v5", "HalfCheetah-v5"]
    root = Path("results/main")
    
    results = []
    for env in envs:
        paths = list((root / env).glob("seed_*.pkl"))
        if not paths: continue
        
        results.append({
            "Env": env,
            "eps_u": get_auroc(paths, "eps_u"),
            "grad_norm": get_auroc(paths, "grad_norm"),
            "exp_var": get_auroc(paths, "explained_var", sign=1), # higher exp_var should be better? no, higher exp_var usually means lower harm. Wait. 
            # harmful if delta_J < 0. 
            # lower exp_var -> higher harm? then sign=1 (lower exp_var is more "harmful")
            "delta_hat": get_auroc(paths, "delta_hat_k")
        })
    
    df = pd.DataFrame(results)
    print(df.to_markdown())

if __name__ == "__main__":
    main()
