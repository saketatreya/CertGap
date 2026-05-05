
import os
import time
import json
import subprocess
from pathlib import Path

ENVS = ["Ant-v5", "Humanoid-v5"]
TOTAL_SEEDS = 10
TARGETS = []
for env in ENVS:
    ts = 300000 if "Ant" in env else 500000
    for s in range(TOTAL_SEEDS):
        TARGETS.append({
            "env": env,
            "seed": s,
            "total": ts,
            "pkl": Path(f"results/sac/{env}/seed_{s}.pkl"),
            "hb": Path(f"results/sac/{env}/seed_{s}.heartbeat")
        })

def get_hb_progress(path):
    try:
        if path.exists():
            with open(path, "r") as f:
                data = json.load(f)
                return data.get("timesteps", 0), data.get("total", 1)
    except:
        pass
    return 0, 1

def count_workers():
    try:
        out = subprocess.check_output(["ps", "-A", "-o", "command"])
        return out.decode().count("multiprocessing.spawn.spawn_main")
    except:
        return 0

def main():
    start_time = time.time()
    
    try:
        while True:
            os.system("clear")
            workers = count_workers()
            done_count = 0
            total_progress = 0
            
            active_lines = []
            
            for t in TARGETS:
                if t["pkl"].exists():
                    done_count += 1
                    total_progress += 1.0
                elif t["hb"].exists():
                    # check if heartbeat is stale (older than 15 mins)
                    # Humanoid is slow, especially with 6 workers.
                    if time.time() - t["hb"].stat().st_mtime < 900:
                        cur, tot = get_hb_progress(t["hb"])
                        p = cur / tot
                        total_progress += p
                        active_lines.append(f"  {t['env']} Seed {t['seed']}: {cur//1000}k/{tot//1000}k ({p*100:.1f}%)")
                    else:
                        # stale heartbeat
                        pass
                else:
                    pass
            
            avg_progress = total_progress / len(TARGETS)
            elapsed = time.time() - start_time
            if avg_progress > 0:
                eta_min = (elapsed / avg_progress - elapsed) / 60
            else:
                eta_min = -1

            print(f"SAC SWEEP MONITOR | Workers: {workers} | Done: {done_count}/{len(TARGETS)}")
            print(f"Overall Progress: {avg_progress*100:.1f}% | Elapsed: {elapsed/60:.1f}m | ETA: {eta_min:.1f}m")
            print("-" * 50)
            
            if active_lines:
                print("Active Seeds:")
                for line in active_lines[:10]: # show first 10
                    print(line)
                if len(active_lines) > 10:
                    print(f"  ... and {len(active_lines)-10} more")
            else:
                print("No active heartbeats detected in the last 15 mins.")
                print("Workers may be processing slow environments like Humanoid.")
            
            if done_count >= len(TARGETS):
                print("\nAll runs completed!")
                break
                
            time.sleep(5)
    except KeyboardInterrupt:
        print("\nMonitoring stopped.")

if __name__ == "__main__":
    main()
