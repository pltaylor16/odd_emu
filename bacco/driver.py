# === run_make_data_driver.py ===

import os
import subprocess


# --- Settings ---
n_jobs = 100
n_procs_per_job = 30
n_total_samples = 120000
samples_per_rank = n_total_samples // (n_jobs * n_procs_per_job)
base_output_dir = "/srv/scratch3/taylor.4264/odd_emu/production_run_bacco"

# --- Launch jobs ---
for i in range(n_jobs):
    run_dir = os.path.join(base_output_dir, f"run_{i:03d}")
    os.makedirs(run_dir, exist_ok=True)
    
    cmd = f"mpirun -n {n_procs_per_job} python make_data.py {run_dir} {i} {n_jobs * n_procs_per_job} {samples_per_rank}"
    print(f"Launching: {cmd}")
    p = subprocess.run(cmd, shell=True)




