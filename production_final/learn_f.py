import os
import sys
import jax
import jax.numpy as jnp
import numpy as np
import equinox as eqx
import optax

# --- Parse arguments ---
z_idx = int(sys.argv[1])
gpu_id = sys.argv[2]
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

# --- Paths ---
save_path = "/srv/scratch2/taylor.4264/odd_emu/production_models_final"
parent_dir = "/srv/scratch2/taylor.4264/odd_emu/production_run_final/merged/"
os.makedirs(save_path, exist_ok=True)

k_shape = 182

# --- Model definition ---
class RHS(eqx.Module):
    mlp: eqx.nn.MLP
    def __init__(self, key):
        self.mlp = eqx.nn.MLP(in_size=k_shape+3, out_size=k_shape, width_size=512, depth=4, key=key)

    def __call__(self, P, H, rho, z):
        x = jnp.concatenate([P, H, rho, z])
        return self.mlp(x)

@eqx.filter_value_and_grad
def loss_fn(model_params, model, X_P, X_H, X_rho, X_z, y):
    model = eqx.combine(model_params, model)
    def single_example(p, h, r, z, y_true):
        y_pred = model(p, h, r, z)
        return jnp.mean((y_pred - y_true) ** 2)
    losses = jax.vmap(single_example)(X_P, X_H, X_rho, X_z, y)
    return jnp.mean(losses)

@eqx.filter_jit
def step(model_params, model, opt_state, X_P, X_H, X_rho, X_z, y):
    loss, grads = loss_fn(model_params, model, X_P, X_H, X_rho, X_z, y)
    updates, opt_state = opt.update(grads, opt_state)
    model_params = optax.apply_updates(model_params, updates)
    return model_params, opt_state, loss

# --- Load data ---
Hz_all = jnp.load(parent_dir + f"Hz_{z_idx}.npy").astype(jnp.float32)
pk_all = jnp.load(parent_dir + f"pk_nl_{z_idx}.npy").astype(jnp.float32)
z_grid = jnp.load(parent_dir + f"z_{z_idx}.npy")
rho_m_all = jnp.load(parent_dir + f"rho_m_{z_idx}.npy").astype(jnp.float32)  # kg/m^3

# --- Normalize H(z) ---
H_flat = Hz_all.reshape(-1)
H_mean = jnp.mean(H_flat)
H_std = jnp.std(H_flat)
Hz_all = (Hz_all - H_mean) / H_std

# --- Normalize rho_m(z) ---
rho_flat = rho_m_all.reshape(-1)
log_rho_flat = jnp.log10(rho_flat + 1e-30)
log_rho_mean = jnp.mean(log_rho_flat)
log_rho_std = jnp.std(log_rho_flat)
log_rho_all = (jnp.log10(rho_m_all + 1e-30) - log_rho_mean) / log_rho_std

# --- Compute derivatives ---
dz = jnp.diff(z_grid)
pk_log = jnp.log1p(pk_all)
pk_diff = pk_log[:, 1:, :] - pk_log[:, :-1, :]
dlogpk_dz = pk_diff / dz[None, :, None]

# --- Prepare input data ---
P_input = pk_log[:, :-1, :]
H_input = Hz_all[:, :-1]
rho_input = log_rho_all[:, :-1]
z_input = jnp.broadcast_to(z_grid[:-1][None, :], H_input.shape)

N = P_input.shape[0] * P_input.shape[1]
X_P = P_input.reshape(N, k_shape)
X_H = H_input.reshape(N, 1)
X_rho = rho_input.reshape(N, 1)
X_z = z_input.reshape(N, 1)
y = dlogpk_dz.reshape(N, k_shape)

# --- Split data ---
split_idx = int(0.9 * N)
X_P_train, X_P_val = X_P[:split_idx], X_P[split_idx:]
X_H_train, X_H_val = X_H[:split_idx], X_H[split_idx:]
X_rho_train, X_rho_val = X_rho[:split_idx], X_rho[split_idx:]
X_z_train, X_z_val = X_z[:split_idx], X_z[split_idx:]
y_train, y_val = y[:split_idx], y[split_idx:]

# --- Train model ---
key = jax.random.PRNGKey(z_idx + 5)
model = RHS(key)
model_params = eqx.filter(model, eqx.is_inexact_array)
opt = optax.adam(1e-3)
opt_state = opt.init(model_params)

best_val_loss = jnp.inf
best_model_params = None
patience = 20
wait = 0
max_epochs = 1000
batch_size = 15000
num_batches = split_idx // batch_size
rng = jax.random.PRNGKey(z_idx + 1000)

for epoch in range(max_epochs):
    perm = jax.random.permutation(rng, split_idx)
    X_P_train = X_P_train[perm]
    X_H_train = X_H_train[perm]
    X_rho_train = X_rho_train[perm]
    X_z_train = X_z_train[perm]
    y_train = y_train[perm]

    epoch_loss = 0.0
    for i in range(num_batches):
        start = i * batch_size
        end = start + batch_size
        X_P_batch = X_P_train[start:end]
        X_H_batch = X_H_train[start:end]
        X_rho_batch = X_rho_train[start:end]
        X_z_batch = X_z_train[start:end]
        y_batch = y_train[start:end]
        model_params, opt_state, batch_loss = step(
            model_params, model, opt_state, X_P_batch, X_H_batch, X_rho_batch, X_z_batch, y_batch
        )
        epoch_loss += batch_loss

    epoch_loss /= num_batches
    val_loss, _ = loss_fn(model_params, model, X_P_val, X_H_val, X_rho_val, X_z_val, y_val)

    if epoch % 10 == 0:
        print(f"[Chunk {z_idx}] Epoch {epoch}: Train Loss = {epoch_loss:.6e}, Val Loss = {val_loss:.6e}")

    if val_loss < best_val_loss - 1e-6:
        best_val_loss = val_loss
        best_model_params = model_params
        wait = 0
    else:
        wait += 1
        if wait >= patience:
            print(f"[Chunk {z_idx}] Early stopping at epoch {epoch}. Best Val Loss = {best_val_loss:.6e}")
            break

# --- Save model ---
model_file = os.path.join(save_path, f"learned_model_rhom_zchunk_{z_idx:02d}.eqx")
eqx.tree_serialise_leaves(model_file, best_model_params)
print(f"[Chunk {z_idx}] Saved best model to {model_file}")