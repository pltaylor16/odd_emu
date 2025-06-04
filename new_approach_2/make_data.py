import os
os.environ["JAX_PLATFORM_NAME"] = "cpu"


import numpy as np
import sys
from cosmopower_jax.cosmopower_jax import CosmoPowerJAX
import euclidemu2 as ee2

# --- Parse arguments ---
output_dir = sys.argv[1]
rank = int(sys.argv[2])
n_ranks = int(sys.argv[3])
n_local = int(sys.argv[4])

print(n_local)

# --- Constants and setup ---
emulator = CosmoPowerJAX(probe="mpk_lin")
k_emulator = emulator.modes.astype(np.float32)
k_idx = np.where((k_emulator > 1e-2) & (k_emulator < 4.9))[0]
k = k_emulator[k_idx]
k_shape = k.shape[0]

z_array = np.linspace(0.01, 5.0, 100).astype(np.float32)
nz = len(z_array)

G = 4.30091e-9  # Mpc Msun^-1 (km/s)^2

# --- Storage arrays ---
logpk = np.zeros((n_local, nz, k_shape), dtype=np.float32)
logpk_dz = np.zeros_like(logpk)
Hz = np.zeros((n_local, nz), dtype=np.float32)
rho_m = np.zeros((n_local, nz), dtype=np.float32)
cosmo_all = []

# --- Main loop ---
for i in range(n_local):
    print(f"[rank {rank}] Sample {i}")

    rng = np.random.default_rng(seed=rank * n_local + i)
    As = rng.uniform(1.7e-9, 2.5e-9)
    ns = rng.uniform(0.92, 1.00)
    Omb = rng.uniform(0.04, 0.06)
    Omm = rng.uniform(0.24, 0.40)
    h = rng.uniform(0.61, 0.73)

    cosmo_dict = {'As': As, 'ns': ns, 'Omb': Omb, 'Omm': Omm, 'h': h, 'mnu': 0.0, 'w': -1.0, 'wa': 0.0}
    omega_b = Omb * h**2
    omega_cdm = (Omm - Omb) * h**2
    ln_10_A_s = np.log(1e10 * As)
    cosmo_params = np.array([omega_b, omega_cdm, h, ns, ln_10_A_s], dtype=np.float32)

    # Compute nonlinear log P(k, z)
    for j, z in enumerate(z_array):
        inp = np.append(cosmo_params, z)
        pk_lin = emulator.predict(inp)[k_idx]
        _, boost = ee2.get_boost(cosmo_dict, np.array([z]), k)
        pk_nl = pk_lin * boost[0]
        logpk[i, j] = np.log(pk_nl + 1e-30)

        Hz[i, j] = 100.0 * h * np.sqrt(Omm * (1 + z)**3 + (1 - Omm))
        rho_crit = 3 * Hz[i, j]**2 / (8 * np.pi * G)
        rho_m[i, j] = Omm * rho_crit

    # Finite difference dlogP/dz
    delta_z = 0.0001
    for j, z in enumerate(z_array):
        z_minus = z - delta_z
        z_plus = z + delta_z

        inp_minus = np.append(cosmo_params, z_minus)
        inp_plus = np.append(cosmo_params, z_plus)

        pk_lin_minus = emulator.predict(inp_minus)[k_idx]
        pk_lin_plus = emulator.predict(inp_plus)[k_idx]

        _, boost_minus = ee2.get_boost(cosmo_dict, np.array([z_minus]), k)
        _, boost_plus = ee2.get_boost(cosmo_dict, np.array([z_plus]), k)

        pk_nl_minus = pk_lin_minus * boost_minus[0]
        pk_nl_plus = pk_lin_plus * boost_plus[0]

        logpk_minus = np.log(pk_nl_minus + 1e-30)
        logpk_plus = np.log(pk_nl_plus + 1e-30)
        logpk_dz[i, j] = (logpk_plus - logpk_minus) / (2 * delta_z)

    cosmo_all.append(np.array([As, ns, Omb, Omm, h], dtype=np.float32))

# --- Save ---
os.makedirs(output_dir, exist_ok=True)
np.savez_compressed(
    os.path.join(output_dir, f"rank{rank:03d}.npz"),
    logpk=logpk,
    logpk_dz=logpk_dz,
    Hz=Hz,
    rho_m=rho_m,
    z=z_array,
    k=k,
    cosmology=np.array(cosmo_all)
)

print(f"[rank {rank}] Done. Saved to {output_dir}")