import json
import time
import os
from pathlib import Path

def monitor():
    heartbeat_dir = Path("results/.heartbeats")
    status_path = Path("results/run_status_ppo_intervention_check.json")
    
    print("\033[2J\033[H", end="") # Clear screen
    print("=== CertGap Reproduction Monitor ===")
    
    while True:
        hb_files = list(heartbeat_dir.glob("*.json"))
        active = []
        for hb in hb_files:
            try:
                data = json.loads(hb.read_text())
                pct = 100 * data['steps'] / data['total_timesteps']
                active.append(f"  \033[94m*\033[0m {data['env_id']} (seed {data['seed']}): {pct:4.1f}%")
            except: pass
            
        print("\033[H", end="")
        print(f"=== CertGap Reproduction Monitor [{time.strftime('%H:%M:%S')}] ===")
        print(f"\nActive Runs: {len(active)}")
        if active:
            print("\n".join(active))
        else:
            print("  (No active heartbeats)")
            
        if status_path.exists():
            try:
                status = json.loads(status_path.read_text())
                runs = status.get('runs', [])
                done = sum(1 for r in runs if r.get('rc') == 0)
                failed = sum(1 for r in runs if r.get('rc') != 0)
                total = 10 # Intervention check is 10 seeds
                print(f"\nIntervention Study Status: {done}/{total} done, {failed} failed")
                if done + failed >= total:
                    print("\n\033[92mCOMPLETE: All runs finished.\033[0m")
                    break
            except: pass
            
        time.sleep(2)

if __name__ == "__main__":
    monitor()
