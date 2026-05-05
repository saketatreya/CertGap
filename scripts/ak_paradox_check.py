
import pickle
from pathlib import Path
import numpy as np
from certgap.analysis.harm_prediction import auroc

def load_seeds(folder: Path) -> list[dict]:
    out = []
    for p in sorted(Path(folder).glob("seed_*.pkl")):
        with open(p, "rb") as f:
            d = pickle.load(f)
        if d["metadata"]["n_updates"] < 10:
            continue
        out.append(d)
    return out

def per_seed_metrics(seeds: list[dict]):
    ak_imp_aurocs = [] # predicting improvement (high Ak -> harm=0)
    dh_imp_aurocs = [] # predicting improvement (low Delta -> harm=0, so we use -Delta)
    n_episodes_list = []
    for d in seeds:
        log = d["log"]
        h = np.asarray(log["harmful_k"], dtype=float)
        
        # Ak AUROC for predicting improvement (higher Ak -> beneficial)
        if "A_k" in log:
            m_ak = np.asarray(log["A_k"], dtype=float)
            ak_imp_aurocs.append(auroc(m_ak, h))
        
        # -Delta AUROC for predicting improvement (lower Delta -> beneficial)
        if "delta_hat_k" in log:
            m_dh = -np.asarray(log["delta_hat_k"], dtype=float)
            dh_imp_aurocs.append(auroc(m_dh, h))

        if "n_episodes" in log:
            n_episodes_list.append(np.mean(log["n_episodes"]))
            
    return ak_imp_aurocs, dh_imp_aurocs, n_episodes_list

ENVS = [
    "LunarLander-v3", "CartPole-v1", "Acrobot-v1",
    "Hopper-v5", "HalfCheetah-v5", "Walker2d-v5", "Ant-v5", "Humanoid-v5",
]

def main():
    env_results = {}
    for env in ENVS:
        seeds = load_seeds(Path(f"results/main/{env}"))
        if not seeds: continue
        ak_imp, dh_imp, n_ep = per_seed_metrics(seeds)
        env_results[env] = {
            "ak": np.nanmedian(ak_imp),
            "dh": np.nanmedian(dh_imp),
            "n_ep": np.nanmedian(n_ep)
        }
        print(f"{env:<15} | n_ep: {env_results[env]['n_ep']:>6.2f} | Ak AUROC (Improvement): {env_results[env]['ak']:>6.3f} | -Delta AUROC (Improvement): {env_results[env]['dh']:>6.3f}")

    # Correlations
    envs = list(env_results.keys())
    n_eps = [env_results[e]["n_ep"] for e in envs]
    aks = [env_results[e]["ak"] for e in envs]
    dhs = [env_results[e]["dh"] for e in envs]
    
    ak_corr = np.corrcoef(n_eps, aks)[0, 1]
    dh_corr = np.corrcoef(n_eps, dhs)[0, 1]
    
    print("\nCorrelations with episode density:")
    print(f"Ak AUROC vs n_ep: {ak_corr:>6.3f}")
    print(f"Delta AUROC vs n_ep: {dh_corr:>6.3f}")

if __name__ == "__main__":
    main()
