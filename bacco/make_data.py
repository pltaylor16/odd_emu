# make_data_bacco.py
import os
import numpy as np
import sys
from pathlib import Path
from mpi4py import MPI
import baccoemu

# --- Constants ---
G = 4.30091e-9  # Mpc Msun^-1 (km/s)^2
z_min, z_max, nz = 0.01, 1.5, 20
z_array = np.linspace(z_min, z_max, nz)
dz = 0.0001

# --- MPI setup ---
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# --- Parse command-line args ---
output_dir = sys.argv[1]
n_local = int(sys.argv[4])  # number of samples per rank

# --- BACCO emulator and k setup ---
emulator = baccoemu.Matter_powerspectrum()
k_vals = np.logspace(-2, np.log10(emulator.emulator['nonlinear']['k'].max()), 30)

# --- Allocate output arrays ---
logpk = np.zeros((n_local, nz, len(k_vals)), dtype=np.float32)
logpk_dz = np.zeros_like(logpk)
Hz = np.zeros((n_local, nz), dtype=np.float32)
rho_m = np.zeros((n_local, nz), dtype=np.float32)
cosmo_all = []

# --- Loop over samples ---
for i in range(n_local):
    print(f"[Rank {rank}] Sample {i+1}/{n_local}")
    rng = np.random.default_rng(seed=rank * n_local + i)

    # Sample parameters from priors
    sigma8 = rng.uniform(0.73, 0.9)
    Omega_m = rng.uniform(0.23, 0.4)
    Omega_b = rng.uniform(0.04, 0.06)
    ns = rng.uniform(0.92, 1.01)
    h = rng.uniform(0.6, 0.8)

    #omega_b = Omega_b * h**2
    #omega_c = (Omega_m - Omega_b) * h**2

    params = {
        'omega_cold'    : Omega_m,
        'sigma8_cold'   : sigma8,
        'omega_baryon'  : Omega_b,
        'ns'            : ns,
        'hubble'        : h,
        'neutrino_mass' : 0.0,
        'w0'            : -1.0,
        'wa'            : 0.0,
    }

    cosmo_all.append([sigma8, Omega_m, Omega_b, ns, h])

    # H0 and critical density
    H0 = 100.0 * h
    rho_crit_0 = 3 * H0**2 / (8 * np.pi * G)

    for j, z in enumerate(z_array):
        a = 1.0 / (1.0 + z)
        params["expfactor"] = a

        # Nonlinear P(k)
        k_bacco, pk_nl = emulator.get_nonlinear_pk(k=k_vals, cold=False, **params)
        logpk[i, j] = np.log(pk_nl + 1e-30)

        # Derivative
        z_minus = max(z - dz, z_min)
        z_plus = min(z + dz, z_max)
        a_minus = 1.0 / (1.0 + z_minus)
        a_plus = 1.0 / (1.0 + z_plus)

        params["expfactor"] = a_minus
        _, pk_minus = emulator.get_nonlinear_pk(k=k_vals, cold=False, **params)
        params["expfactor"] = a_plus
        _, pk_plus = emulator.get_nonlinear_pk(k=k_vals, cold=False, **params)
        logpk_dz[i, j] = (np.log(pk_plus + 1e-30) - np.log(pk_minus + 1e-30)) / (z_plus - z_minus)

        # Hz from Friedmann equation (w=-1)
        Ez2 = Omega_m * (1 + z)**3 + (1 - Omega_m)
        Hz[i, j] = H0 * np.sqrt(Ez2)

        # rho_m
        rho_m[i, j] = Omega_m * rho_crit_0 * (1 + z)**3

# --- Save ---
Path(output_dir).mkdir(parents=True, exist_ok=True)
np.savez_compressed(
    os.path.join(output_dir, f"rank{rank:03d}.npz"),
    logpk=logpk,
    logpk_dz=logpk_dz,
    Hz=Hz,
    rho_m=rho_m,
    z=z_array.astype(np.float32),
    k=k_vals,
    cosmology=np.array(cosmo_all)
)

print(f"[Rank {rank}] Finished and saved {n_local} samples to {output_dir}")