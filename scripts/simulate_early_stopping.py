"""Simulate an early-stopping rule based on start-state bias."""

import pickle
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def collect_data():
    all_delta_hat = []
    all_delta_J = []
    
    # Use Humanoid as it has the most updates and interesting dynamics
    root = Path("results/main/Humanoid-v5")
    for p in root.glob("seed_*.pkl"):
        with p.open("rb") as f:
            log = pickle.load(f)["log"]
            dh = log["delta_hat_k"]
            dj = log["delta_J_k"]
            # Filter NaNs (last update)
            mask = np.isfinite(dh) & np.isfinite(dj)
            all_delta_hat.extend(dh[mask])
            all_delta_J.extend(dj[mask])
            
    return np.array(all_delta_hat), np.array(all_delta_J)

def simulate():
    dh, dj = collect_data()
    
    # Normalize dh if needed, but let's use raw for now
    thresholds = np.linspace(np.percentile(dh, 5), np.percentile(dh, 95), 50)
    
    results = []
    for t in thresholds:
        # Policy: If delta_hat > t, stop update (delta_J = 0)
        # Else, keep update (delta_J = dj)
        effective_dj = np.where(dh > t, 0, dj)
        
        harmful_updates = (dj < 0).sum()
        harm_prevented = ((dj < 0) & (dh > t)).sum()
        
        improvement_lost = dj[(dj > 0) & (dh > t)].sum()
        total_improvement = dj[dj > 0].sum()
        
        results.append({
            "threshold": t,
            "harm_prevented_frac": harm_prevented / harmful_updates,
            "improvement_kept_frac": 1.0 - (improvement_lost / total_improvement),
            "net_gain": effective_dj.sum() - dj.sum()
        })
        
    return pd.DataFrame(results)

if __name__ == "__main__":
    df = simulate()
    
    plt.figure(figsize=(8, 6))
    plt.plot(df["harm_prevented_frac"], df["improvement_kept_frac"], "o-", markersize=4)
    plt.xlabel("Fraction of Harmful Updates Prevented")
    plt.ylabel("Fraction of Beneficial Improvement Retained")
    plt.title("Practical Prescription: Early Stopping via Start-State Bias\n(Humanoid-v5 Audit)")
    plt.grid(True, alpha=0.3)
    
    # Mark some points
    for i in [10, 25, 40]:
        row = df.iloc[i]
        plt.annotate(f"dh > {row['threshold']:.1f}", (row['harm_prevented_frac'], row['improvement_kept_frac']),
                     textcoords="offset points", xytext=(0,10), ha='center', fontsize=8)
    
    plt.savefig("figures/out/fig_early_stopping_pareto.pdf")
    print("wrote figures/out/fig_early_stopping_pareto.pdf")
    
    # Print best net gain
    best = df.loc[df["net_gain"].idxmax()]
    print(f"Best threshold: {best['threshold']:.2f}")
    print(f"Harm prevented: {best['harm_prevented_frac']:.1%}")
    print(f"Improvement kept: {best['improvement_kept_frac']:.1%}")
    print(f"Net return gain: {best['net_gain']:.2f}")
