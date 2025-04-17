import re
import os
import pandas as pd
from pathlib import Path
from datetime import datetime

def parse_moore_log(log_path, output_dir="moore_csv_out"):
    with open(log_path, 'r') as f:
        lines = f.readlines()

    # Regular expressions
    time_pattern = re.compile(r'^(\d{2}/\d{2}/\d{4} \d{2}:\d{2}:\d{2})')
    data_pattern = re.compile(
        r'\[INFO\] Epoch (\d+) \| EnvName: (.*?) .*?AverageReturn: ([0-9.]+)'
    )

    task_data = {}
    start_time = None

    for line in lines:
        # Parse timestamp
        time_match = time_pattern.match(line)
        if time_match:
            timestamp = datetime.strptime(time_match.group(1), "%d/%m/%Y %H:%M:%S")
            if start_time is None:
                start_time = timestamp
            wall_time = (timestamp - start_time).total_seconds()
        else:
            wall_time = None  # fallback

        # Parse epoch and return
        match = data_pattern.search(line)
        if match:
            epoch = int(match.group(1))
            env_name = match.group(2)
            avg_return = float(match.group(3))

            if env_name not in task_data:
                task_data[env_name] = []

            task_data[env_name].append({
                "wall_time": wall_time,
                "step": epoch,  # Keeping 'step' as epoch number for alignment
                "value": avg_return,
            })

    # Output
    Path(output_dir).mkdir(exist_ok=True)
    for env_name, records in task_data.items():
        df = pd.DataFrame(records)
        filename = env_name.replace("-", "_") + ".csv"
        df.to_csv(Path(output_dir) / filename, index=False)
        print(f"Saved: {filename}")

# Example usage
if __name__ == "__main__":
    log_file = "logs/minigrid/MT/MT3/ppo_mt_moore_multihead_2e/seed_0/seed_0.log"  # ← change to your actual log file path
    parse_moore_log(log_file)