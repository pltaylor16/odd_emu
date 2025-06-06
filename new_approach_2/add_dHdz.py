# add_Hz_derivative.py

import numpy as np

# --- Load merged data ---
data_path = "/srv/scratch3/taylor.4264/odd_emu/production_run_logpk/merged/logpk_data.npz"
data = np.load(data_path)

Hz = data["Hz"]
cosmo = data["cosmology"]
z = data["z"]
k = data["k"]

n_samples = cosmo.shape[0]
nz = len(z)
dHz_dz = np.zeros((n_samples, nz), dtype=np.float32)

# --- Physical constant ---
G = 4.30091e-9  # Mpc⋅Msun⁻¹⋅(km/s)²

# --- Compute dHz/dz per sample ---
for i in range(n_samples):
    h = cosmo[i, 4]
    Omm = cosmo[i, 3]
    w0 = -1.0
    wa = 0.0

    H0 = 100.0 * h
    a = 1.0 / (1.0 + z)
    Ez2_matter = Omm * (1 + z) ** 3
    Ez2_de = (1 - Omm) * (1 + z) ** (3 * (1 + w0 + wa)) * np.exp(-3 * wa * z / (1 + z))
    Ez2 = Ez2_matter + Ez2_de
    E = np.sqrt(Ez2)

    dEz2_matter_dz = 3 * Omm * (1 + z) ** 2
    dEz2_de_dz = (1 - Omm) * (
        3 * (1 + w0 + wa) * (1 + z) ** (3 * (1 + w0 + wa) - 1) * np.exp(-3 * wa * z / (1 + z)) +
        (1 + z) ** (3 * (1 + w0 + wa)) * np.exp(-3 * wa * z / (1 + z)) * (3 * wa) / (1 + z) ** 2
    )
    dEz2_dz = dEz2_matter_dz + dEz2_de_dz
    dHz_dz[i] = 0.5 * H0 * dEz2_dz / E

# --- Save updated file ---
out_path = data_path.replace(".npz", "_with_dHz_dz.npz")
np.savez_compressed(
    out_path,
    logpk=data["logpk"],
    logpk_dz=data["logpk_dz"],
    Hz=Hz,
    dHz_dz=dHz_dz,
    rho_m=data["rho_m"],
    cosmology=cosmo,
    z=z,
    k=k
)

print(f"[DONE] Saved updated file with dHz_dz to: {out_path}")