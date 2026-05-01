import pickle
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
import pandas as pd

def main():
    envs = ["Humanoid-v5", "Hopper-v5", "HalfCheetah-v5", "Walker2d-v5", "Ant-v5", "LunarLander-v3"]
    root = Path("results/main")
    
    corrs = []
    for env in envs:
        all_A = []
        all_dh = []
        paths = list((root / env).glob("seed_*.pkl"))
        for p in paths:
            with open(p, "rb") as f:
                log = pickle.load(f)["log"]
                A = log["A_k"]
                dh = log["delta_hat_k"]
                idx = np.isfinite(A) & np.isfinite(dh)
                all_A.extend(A[idx])
                all_dh.extend(dh[idx])
        
        if all_A:
            rho, _ = spearmanr(all_A, all_dh)
            corrs.append({"Env": env, "rho": rho, "n": len(all_A)})
            
    df = pd.DataFrame(corrs)
    print(df.to_markdown())

if __name__ == "__main__":
    main()
