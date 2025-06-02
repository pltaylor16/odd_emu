# === Driver Script: run_make_lcdm_logpk_deriv.py ===

import os
import subprocess

n_jobs = 10
n_procs_per_job = 20
base_output_dir = "/srv/scratch3/taylor.4264/odd_emu/production_run_logpk"

for i in range(n_jobs):
    run_dir = os.path.join(base_output_dir, f"run_{i:03d}")
    os.makedirs(run_dir, exist_ok=True)
    cmd = f"mpirun -n {n_procs_per_job} python make_lcdm_logpk_deriv.py {run_dir} {i} {n_jobs}"
    print(f"Launching: {cmd}")
    subprocess.run(cmd, shell=True)