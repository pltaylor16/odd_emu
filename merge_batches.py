import os
import numpy as np
from glob import glob

# --- Set base directory containing all batch run subdirectories ---
base_dir = "/srv/scratch2/taylor.4264/odd_emu/batched_high_res"
run_dirs = sorted([d for d in glob(os.path.join(base_dir, "run_*")) if os.path.isdir(d)])

all_params = []
all_Hz = []
all_pk = []

for run_dir in run_dirs:
    print(f"Processing: {run_dir}")
    
    # Load all rank files in that run directory
    param_files = sorted(glob(os.path.join(run_dir, "params_rank*.npy")))
    Hz_files = sorted(glob(os.path.join(run_dir, "Hz_rank*.npy")))
    pk_files = sorted(glob(os.path.join(run_dir, "pk_nl_rank*.npy")))

    for pf, hf, pkf in zip(param_files, Hz_files, pk_files):
        try:
            all_params.append(np.load(pf))
            all_Hz.append(np.load(hf))
            all_pk.append(np.load(pkf))
        except Exception as e:
            print(f"Failed to read {pf}, {hf}, or {pkf}: {e}")

# --- Concatenate ---
params_all = np.concatenate(all_params, axis=0)
Hz_all = np.concatenate(all_Hz, axis=0)
pk_all = np.concatenate(all_pk, axis=0)

# --- Save combined arrays ---
np.save(os.path.join(base_dir, "params_all.npy"), params_all)
np.save(os.path.join(base_dir, "Hz_all.npy"), Hz_all)
np.save(os.path.join(base_dir, "pk_nl_all.npy"), pk_all)

# --- Optionally copy k and z ---
import shutil

for ref_run in run_dirs:
    k_path = os.path.join(ref_run, "k.npy")
    z_path = os.path.join(ref_run, "z.npy")
    if os.path.exists(k_path) and os.path.exists(z_path):
        shutil.copy(k_path, os.path.join(base_dir, "k.npy"))
        shutil.copy(z_path, os.path.join(base_dir, "z.npy"))
        break

print(f"\nMerged {len(params_all)} samples across {len(run_dirs)} runs.")
print(f"Saved merged outputs to: {base_dir}")