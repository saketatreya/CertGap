"""Investigate the Ak Paradox: why is surrogate advantage anti-predictive?"""

import pickle
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import spearmanr

def collect_data():
    all_A = []
    all_dh = []
    all_dj = []
    
    root = Path("results/main/Humanoid-v5")
    for p in root.glob("seed_*.pkl"):
        with p.open("rb") as f:
            log = pickle.load(f)["log"]
            A = log["A_k"]
            dh = log["delta_hat_k"]
            dj = log["delta_J_k"]
            mask = np.isfinite(A) & np.isfinite(dh) & np.isfinite(dj)
            all_A.extend(A[mask])
            all_dh.extend(dh[mask])
            all_dj.extend(dj[mask])
            
    return np.array(all_A), np.array(all_dh), np.array(all_dj)

if __name__ == "__main__":
    A, dh, dj = collect_data()
    
    rho_A_dj, _ = spearmanr(A, dj)
    rho_dh_dj, _ = spearmanr(dh, dj)
    rho_A_dh, _ = spearmanr(A, dh)
    
    print(f"Spearman(Ak, delta_J): {rho_A_dj:.3f}")
    print(f"Spearman(delta_hat, delta_J): {rho_dh_dj:.3f}")
    print(f"Spearman(Ak, delta_hat): {rho_A_dh:.3f}")
    
    # Plot Ak vs delta_hat
    plt.figure(figsize=(8, 6))
    plt.hexbin(A, dh, gridsize=30, cmap='Blues', mincnt=1)
    plt.colorbar(label='Count')
    plt.xlabel("Surrogate Advantage ($A_k$)")
    plt.ylabel("Start-State Bias ($\hat\Delta_k$)")
    plt.title(f"The Overfitting Coupling: $A_k$ vs $\hat\Delta_k$\nHumanoid-v5 Audit (rho={rho_A_dh:.3f})")
    plt.grid(True, alpha=0.3)
    plt.savefig("figures/out/fig_ak_dh_correlation.pdf")
    print("wrote figures/out/fig_ak_dh_correlation.pdf")
    
    # Plot Ak vs delta_J
    plt.figure(figsize=(8, 6))
    plt.hexbin(A, dj, gridsize=30, cmap='Reds', mincnt=1)
    plt.colorbar(label='Count')
    plt.xlabel("Surrogate Advantage ($A_k$)")
    plt.ylabel("Actual Improvement ($\Delta J$)")
    plt.title(f"The Ak Paradox: $A_k$ vs $\Delta J$\nHumanoid-v5 Audit (rho={rho_A_dj:.3f})")
    plt.grid(True, alpha=0.3)
    plt.savefig("figures/out/fig_ak_dj_paradox.pdf")
    print("wrote figures/out/fig_ak_dj_paradox.pdf")
