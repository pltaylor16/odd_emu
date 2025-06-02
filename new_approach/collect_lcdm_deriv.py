# === collect_logpk_deriv.py ===

import os
import numpy as np
from glob import glob

base_dir = "/srv/scratch3/taylor.4264/odd_emu/production_run_logpk"
all_runs = sorted(glob(os.path.join(base_dir, "run_*/rank*.npz")))

all_logpk_dz = []
all_Hz = []
all_rho_m = []
all_z = []
all_logpk = []
cosmo = None

for f in all_runs:
    with np.load(f) as data:
        all_logpk_dz.append(data["logpk_dz"])
        all_Hz.append(data["Hz"])
        all_rho_m.append(data["rho_m"])
        all_z.append(data["z"])
        all_logpk.append(data["logpk"])  # ← NEW
        if cosmo is None:
            cosmo = data["cosmology"]

# Concatenate along redshift axis
logpk_dz_all = np.concatenate(all_logpk_dz, axis=0)
Hz_all = np.concatenate(all_Hz, axis=0)
rho_m_all = np.concatenate(all_rho_m, axis=0)
z_all = np.concatenate(all_z, axis=0)
logpk_all = np.concatenate(all_logpk, axis=0)  # ← NEW

# Verify total number of redshift slices
print(f"Total redshifts collected: {len(z_all)}")

# Save
merged_dir = os.path.join(base_dir, "merged")
os.makedirs(merged_dir, exist_ok=True)

np.save(os.path.join(merged_dir, "logpk_dz.npy"), logpk_dz_all)
np.save(os.path.join(merged_dir, "Hz.npy"), Hz_all)
np.save(os.path.join(merged_dir, "rho_m.npy"), rho_m_all)
np.save(os.path.join(merged_dir, "z.npy"), z_all)
np.save(os.path.join(merged_dir, "logpk.npy"), logpk_all)  # ← NEW
np.save(os.path.join(merged_dir, "cosmo.npy"), cosmo)

# Copy k array
k_path = os.path.join(base_dir, "run_000/k.npy")
k_out_path = os.path.join(merged_dir, "k.npy")
if os.path.exists(k_path):
    import shutil
    shutil.copy(k_path, k_out_path)

print(f"Saved merged files to {merged_dir}")