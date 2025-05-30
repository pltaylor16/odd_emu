import os
os.environ["JAX_PLATFORM_NAME"] = "cpu"

from mpi4py import MPI
from cosmopower_jax.cosmopower_jax import CosmoPowerJAX as CPJ
import euclidemu2 as ee2
import numpy as np
import sys

# --- MPI Setup ---
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# --- Command line argument for save directory ---
save_dir = sys.argv[1]
os.makedirs(save_dir, exist_ok=True)

# --- Construct redshift array list ---
redshift_list = []

z_edges_1 = np.linspace(0.0, 1.0, 21, dtype=np.float32)
for i in range(20):
    redshift_list.append(np.linspace(z_edges_1[i], z_edges_1[i + 1], 100, dtype=np.float32))

z_edges_2 = np.linspace(1.0, 2.0, 21, dtype=np.float32)
for i in range(10):
    redshift_list.append(np.linspace(z_edges_2[i], z_edges_2[i + 1], 100, dtype=np.float32))

z_edges_3 = np.linspace(2.0, 3.5, 21, dtype=np.float32)
for i in range(32):
    redshift_list.append(np.linspace(z_edges_3[i], z_edges_3[i + 1], 100, dtype=np.float32))

print(f"Constructed {len(redshift_list)} redshift arrays.")

# --- Prior ranges ---
prior_bounds = {
    'As': (1.7e-9, 2.5e-9),
    'ns': (0.92, 1.00),
    'Omb': (0.04, 0.06),
    'Omm': (0.24, 0.40),
    'h':  (0.61, 0.73),
}

# --- Emulator setup ---
emulator = CPJ(probe='mpk_lin')
k = emulator.modes.astype(np.float32)
k_idx = np.where((k > 1e-2) & (k < 4.9))
k = k[k_idx]

# --- H(z) computation ---
def compute_Hz(cosmo_par, z_array):
    h = cosmo_par['h']
    H0 = np.float32(100.0 * h)
    Omm = cosmo_par['Omm']
    Ode = 1.0 - Omm
    w0 = cosmo_par.get('w', -1.0)
    wa = cosmo_par.get('wa', 0.0)
    a = 1.0 / (1.0 + z_array)
    Ez_sq = (
        Omm * (1 + z_array)**3 +
        Ode * (a**(-3 * (1 + w0 + wa)) * np.exp(3 * wa * (1 - a)))
    )
    return H0 * np.sqrt(Ez_sq)

# --- Total sample count per rank ---
n_total_samples = 60
samples_per_rank = n_total_samples // size + (rank < n_total_samples % size)
start_idx = rank * (n_total_samples // size) + min(rank, n_total_samples % size)

# --- Generate shared cosmology samples for this rank ---
cosmo_params_list = []
for _ in range(samples_per_rank):
    As  = np.random.uniform(*prior_bounds['As'])
    ns  = np.random.uniform(*prior_bounds['ns'])
    Omb = np.random.uniform(*prior_bounds['Omb'])
    Omm = np.random.uniform(*prior_bounds['Omm'])
    h   = np.random.uniform(*prior_bounds['h'])
    cosmo_params_list.append((As, ns, Omb, Omm, h))

# --- Loop over redshift sets using shared cosmologies ---
for z_idx, redshifts in enumerate(redshift_list):
    n_z = len(redshifts)
    n_k = len(k)

    param_array = np.zeros((samples_per_rank, 5), dtype=np.float32)
    pk_nl_array = np.zeros((samples_per_rank, n_z, n_k), dtype=np.float32)
    Hz_array    = np.zeros((samples_per_rank, n_z), dtype=np.float32)

    for i, (As, ns, Omb, Omm, h) in enumerate(cosmo_params_list):
        try:
            omega_b   = Omb * h**2
            omega_cdm = (Omm - Omb) * h**2
            ln_10_A_s = np.log(1e10 * As)

            param_array[i] = [As, ns, Omb, Omm, h]

            cosmo_par = {
                'As': As, 'ns': ns, 'Omb': Omb, 'Omm': Omm, 'h': h,
                'mnu': 0.0, 'w': -1.0, 'wa': 0.0,
            }

            Hz_array[i] = compute_Hz(cosmo_par, redshifts).astype(np.float32)

            pk_lin_2d = []
            for z in redshifts:
                cosmo_inputs = np.array([omega_b, omega_cdm, h, ns, ln_10_A_s, z], dtype=np.float32)
                pk_full = emulator.predict(cosmo_inputs)
                pk_lin_2d.append(pk_full[k_idx])
            pk_lin_2d = np.stack(pk_lin_2d, axis=0).astype(np.float32)

            _, boost = ee2.get_boost(cosmo_par, redshifts, k)
            boost_2d = np.stack([boost[z] for z in range(n_z)], axis=0).astype(np.float32)

            pk_nl_array[i] = pk_lin_2d * boost_2d

            print(f"[Rank {rank}] Sample {i+1}/{samples_per_rank} complete for z set {z_idx}.")

        except Exception as e:
            print(f"[Rank {rank}] Sample {i+1} failed for z set {z_idx}: {e}")

    np.save(os.path.join(save_dir, f"params_rank{rank}_z{z_idx}.npy"), param_array)
    np.save(os.path.join(save_dir, f"pk_nl_rank{rank}_z{z_idx}.npy"), pk_nl_array)
    np.save(os.path.join(save_dir, f"Hz_rank{rank}_z{z_idx}.npy"), Hz_array)

    if rank == 0:
        np.save(os.path.join(save_dir, f"k_z{z_idx}.npy"), k)
        np.save(os.path.join(save_dir, f"z_z{z_idx}.npy"), redshifts)

print(f"[Rank {rank}] All z sets saved to {save_dir}")