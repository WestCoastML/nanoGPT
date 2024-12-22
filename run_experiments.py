"""
This script orchestrates multiple experiments (e.g., diamond variants) on
multiple GPUs, but only runs a limited number concurrently based on available GPUs.

When one of the running processes ends, the next experiment is started.
After all experiments have finished, it downloads W&B logs.
Make sure you have W&B credentials set up if you want logs downloaded.
"""

import os
import subprocess
from datetime import datetime
import numpy as np
import wandb
import pandas as pd
import torch

# Import the experiments list from config/scaling_experiments.py
from config.scaling_experiments import experiments
import os, subprocess, wandb, pandas as pd, torch
from datetime import datetime

available_gpus = list(range(torch.cuda.device_count()))
max_concurrency = len(available_gpus)
out_base_dir = "out_tinystories"
os.makedirs(out_base_dir, exist_ok=True)

run_names = []
running_processes = []

for i, exp in enumerate(experiments):
    gpu_id = available_gpus[i % len(available_gpus)]
    run_name = exp["run_name"]  # use the short run name from the config
    run_names.append(run_name)

    out_dir = os.path.join(out_base_dir, run_name)
    os.makedirs(out_dir, exist_ok=True)

    cmd = [
        "python", "train.py", "config/train_tinystories.py",
        f"--model_architecture={exp['model_architecture']}",
        f"--out_dir={out_dir}",
        "--wandb_log=True",
        f"--wandb_run_name={run_name}",
        f"--device=cuda:{gpu_id}",
    ]

    if exp['model_architecture'] in ['diamond', 'unetxformer']:
        cmd.append(f"--shape_variant={exp['shape_variant']}")

    cmd.append(f"--n_dims={exp['n_dims']}")
    cmd.append(f"--max_iters={exp['max_iters']}")
    cmd.append(f"--lr_decay_iters={exp['lr_decay_iters']}")

    with open(os.path.join(out_dir, "config_used.txt"), "w") as f:
        for k, v in exp.items():
            f.write(f"{k}: {v}\n")

    while len(running_processes) >= max_concurrency:
        finished_p = None
        for (p, rn, od) in running_processes:
            ret = p.poll()
            if ret is not None:
                finished_p = (p, rn, od)
                break
        if finished_p is None:
            running_processes[0][0].wait()
            finished_p = running_processes[0]
        running_processes.remove(finished_p)

    print("Launching:", " ".join(cmd))
    p = subprocess.Popen(cmd)
    running_processes.append((p, run_name, out_dir))

for (p, run_name, out_dir) in running_processes:
    p.wait()

# Download W&B logs if needed
api = wandb.Api()
ENTITY = "wcml"
PROJECT = "VSLM"

for run_name in run_names:
    runs = api.runs(f"{ENTITY}/{PROJECT}", filters={"display_name": run_name})
    if len(runs) == 0:
        print(f"No run found in W&B for run_name: {run_name}")
        continue
    run = runs[0]
    history = run.history()
    csv_path = os.path.join(out_base_dir, run_name, "wandb_logs.csv")
    history.to_csv(csv_path, index=False)
    print(f"Downloaded W&B logs for {run_name} to {csv_path}")

print("All experiments completed and W&B logs downloaded.")
