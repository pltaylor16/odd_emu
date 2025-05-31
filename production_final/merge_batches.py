import os
import numpy as np
from glob import glob
from tqdm import tqdm

# --- Base input/output directory ---
base_dir = "/srv/scratch3/taylor.4264/odd_emu/production_run_final"  # change if needed
output_dir = os.path.join(base_dir, "merged")
os.makedirs(output_dir, exist_ok=True)

# --- Number of redshift chunks (from z_z{zidx}.npy) ---
sample_run = os.path.join(base_dir, "run_000")
z_files = sorted(glob(os.path.join(sample_run, "z_z*.npy")))
k_files = sorted(glob(os.path.join(sample_run, "k_z*.npy")))
n_chunks = len(z_files)

# --- Loop over redshift chunks ---
for z_idx in tqdm(range(n_chunks), desc="Merging redshift chunks"):
    z, k = None, None
    Hz_all_list = []
    pk_all_list = []

    for run_idx in range(500):  # loop over all 500 job outputs
        run_dir = os.path.join(base_dir, f"run_{run_idx:03d}")
        
        # Check presence of Hz and pk files for this z chunk
        Hz_files = sorted(glob(os.path.join(run_dir, f"Hz_rank*_z{z_idx}.npy")))
        pk_files = sorted(glob(os.path.join(run_dir, f"pk_nl_rank*_z{z_idx}.npy")))

        if not Hz_files or not pk_files:
            continue

        # Load z and k once
        if z is None:
            z = np.load(os.path.join(run_dir, f"z_z{z_idx}.npy")).astype(np.float32)
            k = np.load(os.path.join(run_dir, f"k_z{z_idx}.npy")).astype(np.float32)

        # Concatenate all Hz and pk files for this run
        for f in Hz_files:
            Hz_all_list.append(np.load(f).astype(np.float32))
        for f in pk_files:
            pk_all_list.append(np.load(f).astype(np.float32))

    # Stack and save merged arrays
    Hz_all = np.concatenate(Hz_all_list, axis=0)   # (N_samples_total, 100)
    pk_all = np.concatenate(pk_all_list, axis=0)   # (N_samples_total, 100, nk)

    np.save(os.path.join(output_dir, f"z_{z_idx}.npy"), z)
    np.save(os.path.join(output_dir, f"k_{z_idx}.npy"), k)
    np.save(os.path.join(output_dir, f"Hz_{z_idx}.npy"), Hz_all)
    np.save(os.path.join(output_dir, f"pk_nl_{z_idx}.npy"), pk_all)

    print(f"✅ z_idx {z_idx}: saved {Hz_all.shape[0]} samples to {output_dir}")