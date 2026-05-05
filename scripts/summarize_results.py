
import os
import pickle
import numpy as np
from pathlib import Path
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

def compute_auroc(harmful, score):
    # Higher score should mean higher probability of harm
    # We use -delta_hat_k and -eps_u
    if len(np.unique(harmful)) < 2:
        return 0.5
    return roc_auc_score(harmful, score)

def get_stats(path_pattern):
    delta_hat_aurocs = []
    eps_u_aurocs = []
    
    paths = list(Path("results").glob(path_pattern))
    for p in paths:
        try:
            with open(p, "rb") as f:
                data = pickle.load(f)
                log = data["log"]
                
                # Filter out NaNs
                harmful = log["harmful_k"]
                mask = ~np.isnan(harmful)
                if not mask.any(): continue
                
                harmful = harmful[mask]
                delta_hat = -log["delta_hat_k"][mask]
                eps_u = -log["eps_u"][mask]
                
                # Check for NaNs in scores
                m_dh = ~np.isnan(delta_hat)
                if m_dh.sum() > 5:
                    delta_hat_aurocs.append(compute_auroc(harmful[m_dh], delta_hat[m_dh]))
                
                m_eu = ~np.isnan(eps_u)
                if m_eu.sum() > 5:
                    eps_u_aurocs.append(compute_auroc(harmful[m_eu], eps_u[m_eu]))
        except:
            pass
            
    if not delta_hat_aurocs:
        return 0.5, 0.5
    return np.median(delta_hat_aurocs), np.median(eps_u_aurocs)

def main():
    print(f"{'Environment':<25} | {'-Δ̂_k AUROC':<12} | {'-ε_u AUROC':<12}")
    print("-" * 55)
    
    # SAC
    for env in ["Hopper-v5", "HalfCheetah-v5", "Walker2d-v5", "Ant-v5"]:
        dh, eu = get_stats(f"sac/{env}/seed_*.pkl")
        print(f"SAC {env:<21} | {dh:11.2f} | {eu:11.2f}")
    
    print("-" * 55)
    # PPO
    for env in ["Hopper-v5", "HalfCheetah-v5", "Walker2d-v5", "Ant-v5", "LunarLander-v3"]:
        dh, eu = get_stats(f"main/{env}/seed_*.pkl")
        print(f"PPO {env:<21} | {dh:11.2f} | {eu:11.2f}")

if __name__ == "__main__":
    main()
