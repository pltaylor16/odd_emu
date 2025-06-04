import os
os.environ["JAX_PLATFORM_NAME"] = "cpu"

import numpy as np
import sys
from pathlib import Path
from cosmopower_jax.cosmopower_jax import CosmoPowerJAX
import euclidemu2 as ee2
from mpi4py import MPI

# --- Constants ---
G = 4.30091e-9  # Mpc Msun^-1 (km/s)^2
z_min, z_max, nz = 0.01, 5.0, 20
z_array = np.linspace(z_min, z_max, nz)
dz = 0.0001

# --- MPI setup ---
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# --- Parse command-line args ---
output_dir = sys.argv[1]
# sys.argv[2] is the job index, ignored here
# sys.argv[3] is total number of ranks
n_local = int(sys.argv[4])  # ✅ Correctly assign number of samples to generate

# --- Emulator and k setup ---
emulator = CosmoPowerJAX(probe="mpk_lin")
k_emulator = emulator.modes.astype(np.float32)
k_idx = np.where((k_emulator > 1e-2) & (k_emulator < 4.9))[0][::6]
k = k_emulator[k_idx]
k_shape = len(k)

# --- Output arrays ---
logpk = np.zeros((n_local, nz, k_shape), dtype=np.float32)
logpk_dz = np.zeros_like(logpk)
Hz = np.zeros((n_local, nz), dtype=np.float32)
rho_m = np.zeros((n_local, nz), dtype=np.float32)
cosmo_all = []

# --- Loop over realizations ---
for i in range(n_local):
    print(f"[Rank {rank}] Sample {i+1}/{n_local}")
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

    # --- P_lin(k, z) ---
    pk_lin_z = np.stack([
        emulator.predict(np.append(cosmo_params, z))[k_idx] for z in z_array
    ])
    
    _, boost_dict = ee2.get_boost(cosmo_dict, z_array, k)
    boost = np.stack([boost_dict[i] for i in range(len(z_array))])
    pk_nl = pk_lin_z * boost
    logpk[i] = np.log(pk_nl + 1e-30)

    # --- Derivatives and background ---
    for j, z in enumerate(z_array):
        z_minus = max(z - dz, z_min)
        z_plus = min(z + dz, z_max)

        pkm = emulator.predict(np.append(cosmo_params, z_minus))[k_idx]
        pkp = emulator.predict(np.append(cosmo_params, z_plus))[k_idx]
        _, bm = ee2.get_boost(cosmo_dict, [z_minus], k)
        _, bp = ee2.get_boost(cosmo_dict, [z_plus], k)

        pk_nl_m = pkm * bm[0]
        pk_nl_p = pkp * bp[0]
        logpk_dz[i, j] = (np.log(pk_nl_p + 1e-30) - np.log(pk_nl_m + 1e-30)) / (z_plus - z_minus)

        Hz[i, j] = 100.0 * h * np.sqrt(Omm * (1 + z)**3 + (1 - Omm))
        rho_crit = 3 * Hz[i, j]**2 / (8 * np.pi * G)
        rho_m[i, j] = Omm * rho_crit

    cosmo_all.append([As, ns, Omb, Omm, h])

# --- Save ---
Path(output_dir).mkdir(parents=True, exist_ok=True)
np.savez_compressed(
    os.path.join(output_dir, f"rank{rank:03d}.npz"),
    logpk=logpk,
    logpk_dz=logpk_dz,
    Hz=Hz,
    rho_m=rho_m,
    z=z_array.astype(np.float32),
    k=k,
    cosmology=np.array(cosmo_all)
)

print(f"Finished rank {rank}, saved {n_local} samples to {output_dir}")