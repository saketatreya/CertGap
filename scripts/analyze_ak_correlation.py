import pickle
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
import pandas as pd
from certgap.analysis.harm_prediction import auroc

def main():
    envs = ["Acrobot-v1", "CartPole-v1", "Humanoid-v5", "HalfCheetah-v5", "Ant-v5", "LunarLander-v3", "Walker2d-v5", "Hopper-v5"]
    root = Path("results/main")
    
    corrs = []
    for env in envs:
        all_A = []
        all_dh = []
        all_harmful = []
        paths = list((root / env).glob("seed_*.pkl"))
        for p in paths:
            with open(p, "rb") as f:
                log = pickle.load(f)["log"]
                A = log["A_k"]
                dh = log["delta_hat_k"]
                if "harmful_k" in log:
                    harmful = log["harmful_k"]
                else:
                    harmful = np.full_like(A, np.nan)
                
                idx = np.isfinite(A) & np.isfinite(dh) & np.isfinite(harmful)
                all_A.extend(A[idx])
                all_dh.extend(dh[idx])
                all_harmful.extend(harmful[idx])
        
        if all_A:
            rho, _ = spearmanr(all_A, all_dh)
            ak_auroc = auroc(np.array(all_A), np.array(all_harmful))
            corrs.append({"Env": env, "Ak AUROC": ak_auroc, "rho": rho, "n": len(all_A)})
            
    df = pd.DataFrame(corrs)
    print(df.to_markdown())

if __name__ == "__main__":
    main()
