# train_model_logpk_new.py

import os
import sys
import jax
import jax.numpy as jnp
import numpy as np
import equinox as eqx
import optax

# --- Parse args ---
run_idx = int(sys.argv[1])
gpu_id = sys.argv[2]
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

# --- Load merged dataset ---
data = np.load('/srv/scratch3/taylor.4264/odd_emu/production_run_logpk/merged/logpk_data.npz')
logpk = jnp.array(data["logpk"])
logpk_dz = jnp.array(data["logpk_dz"])
Hz = jnp.array(data["Hz"])
rho_m = jnp.array(data["rho_m"])
z = jnp.array(data["z"])
k = jnp.array(data["k"])

N, nz, nk = logpk.shape
X_P = logpk.reshape(-1, nk)
y = logpk_dz.reshape(-1, nk)

Hz_norm = (Hz - Hz.mean()) / Hz.std()
rho_log = jnp.log10(rho_m + 1e-30)
rho_norm = (rho_log - rho_log.mean()) / rho_log.std()

X_H = Hz_norm.reshape(-1, 1)
X_rho = rho_norm.reshape(-1, 1)
X_z = jnp.tile(z[None, :], (N, 1)).reshape(-1, 1)

# --- Train/val split ---
n_total = X_P.shape[0]
split_idx = int(0.9 * n_total)
X_P_train, X_P_val = X_P[:split_idx], X_P[split_idx:]
X_H_train, X_H_val = X_H[:split_idx], X_H[split_idx:]
X_rho_train, X_rho_val = X_rho[:split_idx], X_rho[split_idx:]
X_z_train, X_z_val = X_z[:split_idx], X_z[split_idx:]
y_train, y_val = y[:split_idx], y[split_idx:]

# --- MLP ---
class RHS(eqx.Module):
    mlp: eqx.nn.MLP
    def __init__(self, key):
        self.mlp = eqx.nn.MLP(in_size=nk+3, out_size=nk, width_size=512, depth=4, key=key)

    def __call__(self, P, H, rho, z):
        return self.mlp(jnp.concatenate([P, H, rho, z]))

@eqx.filter_value_and_grad
def loss_fn(params, model, X_P, X_H, X_rho, X_z, y):
    model = eqx.combine(params, model)
    y_pred = jax.vmap(model)(X_P, X_H, X_rho, X_z)
    return jnp.mean((y_pred - y)**2)

@eqx.filter_jit
def step(params, model, opt_state, X_P, X_H, X_rho, X_z, y):
    loss, grads = loss_fn(params, model, X_P, X_H, X_rho, X_z, y)
    updates, opt_state = opt.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss

# --- Init model ---
save_path = "/srv/scratch3/taylor.4264/odd_emu/models_final"
os.makedirs(save_path, exist_ok=True)
key = jax.random.PRNGKey(run_idx + 5)
model = RHS(key)
params = eqx.filter(model, eqx.is_inexact_array)
opt = optax.adam(1e-3)
opt_state = opt.init(params)

# --- Training loop ---
best_val_loss = jnp.inf
best_params = None
patience = 20
wait = 0
batch_size = 2048
n_epochs = 1000
n_batches = split_idx // batch_size
rng = jax.random.PRNGKey(run_idx + 1000)

for epoch in range(n_epochs):
    rng, subkey = jax.random.split(rng)
    perm = jax.random.permutation(subkey, split_idx)
    X_P_train = X_P_train[perm]
    X_H_train = X_H_train[perm]
    X_rho_train = X_rho_train[perm]
    X_z_train = X_z_train[perm]
    y_train = y_train[perm]

    epoch_loss = 0.0
    for i in range(n_batches):
        s = i * batch_size
        e = s + batch_size
        batch = (X_P_train[s:e], X_H_train[s:e], X_rho_train[s:e], X_z_train[s:e], y_train[s:e])
        params, opt_state, batch_loss = step(params, model, opt_state, *batch)
        epoch_loss += batch_loss
    epoch_loss /= n_batches

    val_loss, _ = loss_fn(params, model, X_P_val, X_H_val, X_rho_val, X_z_val, y_val)

    if epoch % 10 == 0:
        print(f"Run {run_idx}, Epoch {epoch}: Train Loss = {epoch_loss:.3e}, Val Loss = {val_loss:.3e}")

    if val_loss < best_val_loss - 1e-6:
        best_val_loss = val_loss
        best_params = params
        wait = 0
    else:
        wait += 1
        if wait >= patience:
            print(f"Early stopping at epoch {epoch}, best val loss: {best_val_loss:.3e}")
            break

eqx.tree_serialise_leaves(f"{save_path}/learned_model_logpk_{run_idx}.eqx", best_params)
print(f"Saved model for run {run_idx} to {save_path}")