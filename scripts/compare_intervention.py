import pickle
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

def load_results(folder):
    returns = []
    root = Path(folder)
    for p in root.glob("seed_*.pkl"):
        with open(p, "rb") as f:
            log = pickle.load(f)["log"]
            returns.append(log["J_k"])
    return np.array(returns)

def main():
    vanilla = load_results("results/intervention_check/vanilla")
    gated = load_results("results/intervention_check/gated")
    
    plt.figure(figsize=(10, 6))
    for i, v in enumerate(vanilla):
        plt.plot(v, color='gray', alpha=0.3, label='Vanilla' if i == 0 else "")
    for i, g in enumerate(gated):
        plt.plot(g, color='blue', alpha=0.3, label='Gated' if i == 0 else "")
        
    plt.plot(vanilla.mean(0), color='black', linewidth=2, label='Vanilla (mean)')
    plt.plot(gated.mean(0), color='blue', linewidth=2, label='Gated (mean)')
    
    plt.legend()
    plt.xlabel("Update")
    plt.ylabel("Return")
    plt.title("PPO Intervention: Vanilla vs Gated ($\hat\Delta_k < 161$)")
    plt.savefig("figures/out/fig_intervention_check.pdf")
    print("wrote figures/out/fig_intervention_check.pdf")

if __name__ == "__main__":
    main()
