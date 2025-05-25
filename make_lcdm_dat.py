from mpi4py import MPI
from cosmopower_jax.cosmopower_jax import CosmoPowerJAX as CPJ
import euclidemu2 as ee2
import numpy as np
import os

# --- MPI Setup ---
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# --- H(z) computation ---
def compute_Hz(cosmo_par, z_array):
    """
    Compute H(z) for a flat universe with CPL dark energy.
    """
    h = cosmo_par['h']
    H0 = 100.0 * h  # km/s/Mpc

    Omm = cosmo_par['Omm']
    Ode = 1.0 - Omm  # assuming flat universe
    w0 = cosmo_par.get('w', -1.0)
    wa = cosmo_par.get('wa', 0.0)

    a = 1.0 / (1.0 + z_array)
    Ez_sq = (
        Omm * (1 + z_array)**3 +
        Ode * (a**(-3 * (1 + w0 + wa)) * np.exp(3 * wa * (1 - a)))
    )

    return H0 * np.sqrt(Ez_sq)

# --- Config ---
n_total_samples = 300  # total across all ranks
samples_per_rank = n_total_samples // size + (rank < n_total_samples % size)
start_idx = rank * (n_total_samples // size) + min(rank, n_total_samples % size)

import sys
save_dir = sys.argv[1]
os.makedirs(save_dir, exist_ok=True)
os.makedirs(save_dir, exist_ok=True)

redshifts = np.linspace(0., 5, 500)

# --- Prior ranges ---
prior_bounds = {
    'As': (1.7e-9, 2.5e-9),
    'ns': (0.92, 1.00),
    'Omb': (0.04, 0.06),
    'Omm': (0.24, 0.40),
    'h':  (0.61, 0.73),
}

# --- Emulator and k setup ---
emulator = CPJ(probe='mpk_lin')
k = emulator.modes
idx = np.where((k > 1e-2) & (k < 4.9))
k = k[idx]
n_z = len(redshifts)
n_k = len(k)

# --- Allocate local storage ---
param_array = np.zeros((samples_per_rank, 5))         # [As, ns, Omb, Omm, h]
pk_nl_array = np.zeros((samples_per_rank, n_z, n_k))  # P_nl for each sample
Hz_array    = np.zeros((samples_per_rank, n_z))       # H(z) for each sample

# --- Local sampling loop ---
for i in range(samples_per_rank):
    try:
        # Draw sample
        As  = np.random.uniform(*prior_bounds['As'])
        ns  = np.random.uniform(*prior_bounds['ns'])
        Omb = np.random.uniform(*prior_bounds['Omb'])
        Omm = np.random.uniform(*prior_bounds['Omm'])
        h   = np.random.uniform(*prior_bounds['h'])

        omega_b   = Omb * h**2
        omega_cdm = (Omm - Omb) * h**2
        ln_10_A_s = np.log(1e10 * As)

        param_array[i] = [As, ns, Omb, Omm, h]

        cosmo_par = {
            'As': As,
            'ns': ns,
            'Omb': Omb,
            'Omm': Omm,
            'h': h,
            'mnu': 0.0,
            'w': -1.0,
            'wa': 0.0,
        }

        # Compute H(z)
        Hz_array[i] = compute_Hz(cosmo_par, redshifts)

        # Compute linear power spectrum
        pk_lin_2d = []
        for z in redshifts:
            cosmo_params = np.array([omega_b, omega_cdm, h, ns, ln_10_A_s, z])
            pk_full = emulator.predict(cosmo_params)
            pk_lin_2d.append(pk_full[idx])
        pk_lin_2d = np.stack(pk_lin_2d, axis=0)

        # Compute nonlinear boost
        _, boost = ee2.get_boost(cosmo_par, redshifts, k)
        boost_2d = np.stack([boost[z] for z in range(n_z)], axis=0)

        # Nonlinear P(k,z)
        pk_nl_array[i] = pk_lin_2d * boost_2d

        print(f"[Rank {rank}] Sample {i+1}/{samples_per_rank} complete.")

    except Exception as e:
        print(f"[Rank {rank}] Sample {i+1} failed: {e}")

# --- Save local results ---
np.save(os.path.join(save_dir, f"params_rank{rank}.npy"), param_array)
np.save(os.path.join(save_dir, f"pk_nl_rank{rank}.npy"), pk_nl_array)
np.save(os.path.join(save_dir, f"Hz_rank{rank}.npy"), Hz_array)

# Save shared files only once
if rank == 0:
    np.save(os.path.join(save_dir, "k.npy"), k)
    np.save(os.path.join(save_dir, "z.npy"), redshifts)


print(f"[Rank {rank}] Saved results to {save_dir}")