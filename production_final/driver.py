import os
import subprocess

n_jobs = 500
n_procs_per_job = 40  # match to your `mpirun -n` usage
base_output_dir = "/srv/scratch2/taylor.4264/odd_emu/production_run_final"

for i in range(n_jobs):
    run_dir = os.path.join(base_output_dir, f"run_{i:03d}")
    os.makedirs(run_dir, exist_ok=True)
    
    cmd = f"mpirun -n {n_procs_per_job} python make_lcdm_dat.py {run_dir}"
    print(f"Launching: {cmd}")
    subprocess.run(cmd, shell=True)