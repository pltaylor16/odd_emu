import os
import numpy as np
from glob import glob

# --- Directories ---
base_dir = "/srv/scratch3/taylor.4264/odd_emu/production_run_final"
merged_dir = os.path.join(base_dir, "merged")
os.makedirs(merged_dir, exist_ok=True)

# --- Collect one set of params per rank per run (e.g., z0 only) ---
param_files = sorted(glob(os.path.join(base_dir, "run_*/params_rank*_z0.npy")))

# --- Load and concatenate ---
all_params = []

for f in param_files:
    print(f)
    try:
        arr = np.load(f)
        if arr.ndim == 2 and arr.shape[1] == 5:
            all_params.append(arr)
        else:
            print(f"Skipping malformed or empty: {f}")
    except Exception as e:
        print(f"Failed to read {f}: {e}")

if not all_params:
    raise RuntimeError("No valid parameter files found.")

params_all = np.concatenate(all_params, axis=0)
np.save(os.path.join(merged_dir, "params_all.npy"), params_all.astype(np.float32))

print(f"Saved {params_all.shape[0]} total cosmology samples to {os.path.join(merged_dir, 'params_all.npy')}")