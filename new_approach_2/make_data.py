import numpy as np
import os
import sys
from cosmopower_jax.cosmopower_jax import CosmoPowerJAX
import euclidemu2 as ee2
from scipy.interpolate import InterpolatedUnivariateSpline
from numpy import random, trapezoid

# --- Parse command line args ---
output_dir = sys.argv[1]
rank = int(sys.argv[2])
n_ranks = int(sys.argv[3])
n_local = int(sys.argv[4])  # samples per rank

# --- Constants and setup ---
k_emulator = CosmoPowerJAX(probe="mpk_lin").modes.astype(np.float32)
k_idx = np.where((k_emulator > 1e-2) & (k_emulator < 4.9))[0]
k = k_emulator[k_idx]
k_shape = k.shape[0]


start = rank * n_local
end = (rank + 1) * n_local

# --- Redshift grid ---
z_min, z_max = 0.01, 5.0
nz = 1000
z_array = np.linspace(z_min, z_max, nz)

# --- Boost interpolation grid ---
z_boost_grid = np.linspace(z_min, z_max, 50)

# --- Arrays ---
logpk = np.zeros((n_local, nz, k_shape), dtype=np.float32)
logpk_dz = np.zeros_like(logpk)
Hz = np.zeros((n_local, nz), dtype=np.float32)
rho_m = np.zeros((n_local, nz), dtype=np.float32)
z_out = z_array.astype(np.float32)
cosmo_all = []

# --- Constants ---
G = 4.30091e-9  # Mpc Msun^-1 (km/s)^2

# --- Growth factor ---
def growth_factor(z, Om, w0=-1.0, wa=0.0):
    a_vals = np.atleast_1d(1.0 / (1.0 + z))
    D = []

    def E(a):
        w = w0 + wa * (1 - a)
        Omega_DE = 1.0 - Om
        return np.sqrt(Om / a**3 + Omega_DE * a**(-3 * (1 + w0 + wa)) * np.exp(-3 * wa * (1 - a)))

    for a in a_vals:
        a_grid = np.linspace(a, 1.0, 200)
        integrand = 1.0 / (a_grid * E(a_grid)**3)
        integral = trapezoid(integrand, a_grid)
        pre_factor = 5 * Om * E(a)
        D_growth = pre_factor * integral / 2.0
        D.append(D_growth)

    return D[0] if np.isscalar(z) else np.array(D)

# --- Main loop ---
for i in range(n_local):
    print(i)
    rng = random.default_rng(seed=rank * n_local + i)
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

    pk_lin_z0 = CosmoPowerJAX(probe="mpk_lin").predict(np.append(cosmo_params, 0.0))[k_idx]
    Dz = growth_factor(z_array, Omm)
    pk_lin_z = pk_lin_z0[None, :] * Dz[:, None]**2

    _, boost_grid_dict = ee2.get_boost(cosmo_dict, z_boost_grid, k)
    boost_grid = np.stack([boost_grid_dict[i] for i in range(len(z_boost_grid))])
    boost_interp = [InterpolatedUnivariateSpline(z_boost_grid, boost_grid[:, j], k=3) for j in range(k_shape)]

    # Evaluate interpolated boost * growth for all redshifts in advance
    boost_interp_arr = np.array([interp(z_array) for interp in boost_interp])
    boost_interp_arr *= Dz[None, :]  # apply growth factor

    for j, z in enumerate(z_array):
        Hz[i, j] = 100 * h * np.sqrt(Omm * (1 + z) ** 3 + (1 - Omm))
        rho_crit = 3 * Hz[i, j] ** 2 / (8 * np.pi * G)
        rho_m[i, j] = Omm * rho_crit

        pk_nl = pk_lin_z0 * boost_interp_arr[:, j]
        logpk[i, j] = np.log(pk_nl + 1e-30)

        if j < nz - 1:
            j_minus = max(j - 1, 0)
            j_plus = j + 1
            pk_nl_minus = pk_lin_z0 * boost_interp_arr[:, j_minus]
            pk_nl_plus  = pk_lin_z0 * boost_interp_arr[:, j_plus]
            logpk_dz[i, j] = (np.log(pk_nl_plus + 1e-30) - np.log(pk_nl_minus + 1e-30)) / (z_array[j_plus] - z_array[j_minus])
        else:
            logpk_dz[i, j] = logpk_dz[i, j - 1]

    cosmo_all.append(np.array([As, ns, Omb, Omm, h], dtype=np.float32))

# --- Save ---
os.makedirs(output_dir, exist_ok=True)
np.savez_compressed(
    os.path.join(output_dir, f"rank{rank:03d}.npz"),
    logpk=logpk,
    logpk_dz=logpk_dz,
    Hz=Hz,
    rho_m=rho_m,
    z=z_out,
    k=k,
    cosmology=np.array(cosmo_all)
)

print(f"Finished rank {rank}, saved to {output_dir}")