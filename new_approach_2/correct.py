# correct.py

import numpy as np
import os

# --- Constants ---
G = 4.30091e-9  # Mpc Msun^-1 (km/s)^2


# --- Paths ---
data_path = "/srv/scratch3/taylor.4264/odd_emu/production_run_logpk/merged/logpk_data.npz"

# --- Load ---
data = np.load(data_path)
logpk = data["logpk"]
logpk_dz = data["logpk_dz"]
Hz = data["Hz"]
cosmo = data["cosmology"]
z = data["z"]
k = data["k"]

# --- Recompute rho_m ---
n_samples = cosmo.shape[0]
nz = len(z)
rho_m = np.zeros((n_samples, nz), dtype=np.float32)

for i in range(n_samples):
    Omega_m = cosmo[i, 3]  # Assuming order: [As, ns, Omb, Omm, h]
    h = cosmo[i, 4]
    rho_crit_0 = 3 * (100.*h)**2 / (8 * np.pi * G)
    rho_m[i] = Omega_m * rho_crit_0 * (1 + z)**3

# --- Save corrected file ---
np.savez_compressed(
    data_path,
    logpk=logpk,
    logpk_dz=logpk_dz,
    Hz=Hz,
    rho_m=rho_m,
    cosmology=cosmo,
    z=z,
    k=k
)

print("[DONE] Corrected rho_m saved to", data_path)