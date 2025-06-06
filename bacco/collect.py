# === collect_logpk_data.py ===

import numpy as np
import os
from glob import glob
from tqdm import tqdm

# --- Paths ---
base_dir = "/srv/scratch3/taylor.4264/odd_emu/production_run_bacco"
output_dir = os.path.join(base_dir, "merged")
os.makedirs(output_dir, exist_ok=True)

# --- Gather all npz files ---
all_npz_files = sorted(glob(f"{base_dir}/run_*/rank*.npz"))
print(f"[INFO] Found {len(all_npz_files)} .npz files to merge")

# --- Lists for concatenation ---
logpk_list = []
logpk_dz_list = []
Hz_list = []
rho_m_list = []
cosmo_list = []

# --- Load one file to get z, k ---
sample = np.load(all_npz_files[0])
z = sample["z"]
k = sample["k"]

# --- Loop through files with progress ---
for path in tqdm(all_npz_files, desc="Merging data"):
    data = np.load(path)
    logpk_list.append(data["logpk"])
    logpk_dz_list.append(data["logpk_dz"])
    Hz_list.append(data["Hz"])
    rho_m_list.append(data["rho_m"])
    cosmo_list.append(data["cosmology"])

# --- Concatenate and save ---
np.savez_compressed(
    os.path.join(output_dir, "logpk_data.npz"),
    logpk=np.concatenate(logpk_list, axis=0),
    logpk_dz=np.concatenate(logpk_dz_list, axis=0),
    Hz=np.concatenate(Hz_list, axis=0),
    rho_m=np.concatenate(rho_m_list, axis=0),
    cosmology=np.concatenate(cosmo_list, axis=0),
    z=z,
    k=k
)

print(f"[DONE] Merged data saved to {output_dir}/logpk_data.npz")