# === Script: make_lcdm_logpk_deriv.py ===

import os
os.environ["JAX_PLATFORM_NAME"] = "cpu"

import sys
import numpy as np
from mpi4py import MPI
from cosmopower_jax.cosmopower_jax import CosmoPowerJAX as CPJ
import euclidemu2 as ee2

# --- MPI Setup ---
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# --- Input ---
save_dir = sys.argv[1]
job_idx = int(sys.argv[2])
n_jobs = int(sys.argv[3])
os.makedirs(save_dir, exist_ok=True)

# --- Emulator and grid setup ---
emulator = CPJ(probe='mpk_lin')
k_full = emulator.modes.astype(np.float32)
k_idx = np.where((k_full > 1e-2) & (k_full < 4.9))[0]
k = k_full[k_idx]
nk = len(k)

# --- Prior ranges ---
prior_bounds = {
    'As': (1.7e-9, 2.5e-9),
    'ns': (0.92, 1.00),
    'Omb': (0.04, 0.06),
    'Omm': (0.24, 0.40),
    'h':  (0.61, 0.73),
}

# --- Cosmology sample (shared across all ranks) ---
if rank == 0:
    As  = np.random.uniform(*prior_bounds['As'])
    ns  = np.random.uniform(*prior_bounds['ns'])
    Omb = np.random.uniform(*prior_bounds['Omb'])
    Omm = np.random.uniform(*prior_bounds['Omm'])
    h   = np.random.uniform(*prior_bounds['h'])
    cosmo_array = np.array([As, ns, Omb, Omm, h], dtype=np.float32)
else:
    cosmo_array = np.empty(5, dtype=np.float32)

comm.Bcast(cosmo_array, root=0)
As, ns, Omb, Omm, h = cosmo_array
cosmo_par = {
    'As': As, 'ns': ns, 'Omb': Omb, 'Omm': Omm, 'h': h,
    'mnu': 0.0, 'w': -1.0, 'wa': 0.0,
}

omega_b   = Omb * h**2
omega_cdm = (Omm - Omb) * h**2
ln_10_A_s = np.log(1e10 * As)

# --- Constants for rho_crit ---
G = 4.30091e-9  # Mpc⋅Msun⁻¹⋅(km/s)²

# --- Redshift slice for this job ---
z_full = np.linspace(0.01, 5.0, 20000).astype(np.float32)
z_chunks = np.array_split(z_full, n_jobs)
z_all = z_chunks[job_idx]
nz = len(z_all)

# --- Chunk per rank ---
nz_per_rank = nz // size + (rank < (nz % size))
start = rank * (nz // size) + min(rank, (nz % size))
z_chunk = z_all[start:start + nz_per_rank]

# --- Output arrays ---
logpk_dz_all = np.zeros((len(z_chunk), nk), dtype=np.float32)
logpk_all = np.zeros((len(z_chunk), nk), dtype=np.float32)
Hz_all = np.zeros(len(z_chunk), dtype=np.float32)
rho_m_all = np.zeros(len(z_chunk), dtype=np.float32)

# --- Loop over redshift ---
dz = 0.0001  # central step size

for i, z in enumerate(z_chunk):
    z_minus = max(z - dz, 0.001)
    z_plus = z + dz
    z_cen = z

    def get_pk_nl(zz):
        inp = np.array([omega_b, omega_cdm, h, ns, ln_10_A_s, zz], dtype=np.float32)
        pk_lin = emulator.predict(inp)[k_idx]
        _, boost = ee2.get_boost(cosmo_par, np.array([zz]), k)
        return pk_lin * boost[0]

    pk_minus = get_pk_nl(z_minus)
    pk_plus = get_pk_nl(z_plus)
    pk_cen = get_pk_nl(z_cen)

    # Two-sided central difference for dlogP/dz
    logpk_dz_all[i] = (np.log(pk_plus) - np.log(pk_minus)) / (2 * dz)
    logpk_all[i] = np.log(pk_cen)

    # Background quantities
    a = 1.0 / (1.0 + z)
    Ez_sq = Omm * (1 + z)**3 + (1 - Omm) * a**(-3 * (1 + (-1.0)))
    Hz = 100.0 * h * np.sqrt(Ez_sq)
    rho_crit = 3 * Hz**2 / (8 * np.pi * G)
    Hz_all[i] = Hz
    rho_m_all[i] = Omm * rho_crit

# --- Save output ---
out_path = os.path.join(save_dir, f"rank{rank}.npz")
np.savez_compressed(out_path,
    logpk_dz=logpk_dz_all,
    logpk=logpk_all,
    Hz=Hz_all,
    rho_m=rho_m_all,
    z=z_chunk,
    cosmology=cosmo_array
)

if rank == 0:
    np.save(os.path.join(save_dir, "k.npy"), k)

print(f"[Rank {rank}] Saved {len(z_chunk)} samples to {save_dir}")