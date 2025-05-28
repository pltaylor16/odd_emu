import numpy as np
import os
from glob import glob

base_dir = "/srv/scratch2/taylor.4264/odd_emu/production_run"
merged_dir = os.path.join(base_dir, "merged")
os.makedirs(merged_dir, exist_ok=True)

# --- Constants ---
G = 6.67430e-11  # m^3 kg^-1 s^-2
Mpc_to_m = 3.085677581491367e22  # m/Mpc
H_unit = 100 * 1e3 / Mpc_to_m  # H(z) in s^-1 if given in km/s/Mpc
rho_crit_unit = 3 / (8 * np.pi * G) * H_unit**2  # critical density unit in kg/m^3

# --- Load cosmological parameters (shared across all z chunks) ---
params_path = os.path.join(merged_dir, "params_all.npy")
if not os.path.exists(params_path):
    raise FileNotFoundError(f"{params_path} not found.")
cosmo_all = np.load(params_path)  # shape: (n_cosmos, 5)
Omm = cosmo_all[:, 3]
h = cosmo_all[:, 4]

# --- Loop over redshift chunks and compute ρ_m(z) ---
z_idx = 0
while True:
    hz_path = os.path.join(merged_dir, f"Hz_{z_idx}.npy")
    if not os.path.exists(hz_path):
        break

    Hz_all = np.load(hz_path)  # shape: (n_cosmos, nz)
    Hz_ratio = Hz_all / 100.0
    rho_crit_z = rho_crit_unit * Hz_ratio**2  # (n_cosmos, nz)

    # Compute mean matter density: ρ_m(z) = Ω_m h² × ρ_crit(z)
    rho_m_z = (Omm[:, None] * h[:, None]**2) * rho_crit_z

    outfile = os.path.join(merged_dir, f"rho_m_{z_idx}.npy")
    np.save(outfile, rho_m_z.astype(np.float32))
    print(f"[z chunk {z_idx}] Saved ρ_m(z) to {outfile} with shape {rho_m_z.shape}")

    z_idx += 1