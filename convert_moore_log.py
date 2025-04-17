import os
import numpy as np
import pandas as pd
from pathlib import Path

def convert_npy_to_seed_csv(npy_path, save_dir, epochs_per_seed, wall_time_increment=2.0):
    task_name = os.path.basename(npy_path).replace("_AverageReturn.npy", "")
    flat_returns = np.load(npy_path, allow_pickle=True).flatten()

    # Calculate number of complete seeds
    num_full_seeds = len(flat_returns) // epochs_per_seed
    remainder = len(flat_returns) % epochs_per_seed

    if remainder != 0:
        print(f"⚠️ Warning: trimming {remainder} entries from '{task_name}' (original length = {len(flat_returns)})")

    flat_returns = flat_returns[:num_full_seeds * epochs_per_seed]

    for seed in range(num_full_seeds):
        seed_returns = flat_returns[seed * epochs_per_seed : (seed + 1) * epochs_per_seed]
        df = pd.DataFrame({
            "wall_time": np.arange(epochs_per_seed) * wall_time_increment,
            "step": np.arange(epochs_per_seed),
            "value": seed_returns
        })
        seed_filename = f"{task_name}_seed{seed}.csv"
        df.to_csv(os.path.join(save_dir, seed_filename), index=False)
        print(f"✅ Saved: {seed_filename}")

def batch_convert_npy_dir(npy_dir, save_dir="moore_csv_out", epochs_per_seed=100, wall_time_increment=2.0):
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    for file in os.listdir(npy_dir):
        if file.endswith(".npy") and "_AverageReturn" in file:
            full_path = os.path.join(npy_dir, file)
            convert_npy_to_seed_csv(full_path, save_dir, epochs_per_seed, wall_time_increment)

# === Example usage ===
if __name__ == "__main__":
    npy_dir = "logs/minigrid/MT/MT3/ppo_mt_moore_multihead_2e/"       # ← Update this
    batch_convert_npy_dir(npy_dir, epochs_per_seed=100)