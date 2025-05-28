import numpy as np

# --- Load data ---
params = np.load("/srv/scratch2/taylor.4264/odd_emu/batched/params_all.npy").astype(np.float32)  # shape (N, 5)
z = np.load("/srv/scratch2/taylor.4264/odd_emu/batched/z.npy").astype(np.float32)  # shape (nz,)

# --- Constants ---
G = 6.67430e-11  # m^3 / kg / s^2
Mpc_in_m = 3.08567758128e22  # meters
H0_unit = 100 * 1e3 / Mpc_in_m  # 100 km/s/Mpc in 1/s
rho_crit_prefactor = 3 / (8 * np.pi * G)  # in kg/m^3

# --- Compute rho_crit(z) ---
a = 1.0 / (1.0 + z)
N = params.shape[0]
nz = z.shape[0]
rho_crit = np.zeros((N, nz), dtype=np.float32)

for i in range(N):
    Omm = params[i, 3]
    h = params[i, 4]
    Ode = 1.0 - Omm
    w0 = -1.0
    wa = 0.0

    Ez2 = (
        Omm * (1 + z)**3 +
        Ode * (a**(-3 * (1 + w0 + wa)) * np.exp(3 * wa * (1 - a)))
    )
    H = H0_unit * h * np.sqrt(Ez2)
    rho_crit[i] = rho_crit_prefactor * H**2  # units: kg/m^3

# Optional: convert to Msun/Mpc^3
kg_to_msun = 1.98847e30
rho_crit *= (Mpc_in_m)**3 / kg_to_msun  # now in Msun / Mpc^3

# --- Save result ---
np.save("/srv/scratch2/taylor.4264/odd_emu/batched/rho_crit_all.npy", rho_crit)
print("Saved critical density to rho_crit_all.npy with shape", rho_crit.shape)