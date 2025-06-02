import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use GPU 0

import jax
import jax.numpy as jnp
import numpy as np
import equinox as eqx
import optax

# --- Load Data ---
parent_dir = '/srv/scratch3/taylor.4264/odd_emu/production_run_logpk/merged/'
logpk = jnp.load(parent_dir + "logpk.npy")          # shape (n_samples, n_k)
logpk_dz = jnp.load(parent_dir + "logpk_dz.npy")    # shape (n_samples, n_k)
Hz = jnp.load(parent_dir + "Hz.npy")                # shape (n_samples,)
rho_m = jnp.load(parent_dir + "rho_m.npy")          # shape (n_samples,)
z = jnp.load(parent_dir + "z.npy")                  # shape (n_samples,)

# --- Normalize H and rho ---
H_mean, H_std = jnp.mean(Hz), jnp.std(Hz)
Hz_norm = (Hz - H_mean) / H_std

log_rho = jnp.log10(rho_m + 1e-30)
log_rho_mean, log_rho_std = jnp.mean(log_rho), jnp.std(log_rho)
rho_norm = (log_rho - log_rho_mean) / log_rho_std

# --- Flatten input ---
X_P = logpk.astype(jnp.float32)
X_H = Hz_norm[:, None].astype(jnp.float32)
X_rho = rho_norm[:, None].astype(jnp.float32)
X_z = z[:, None].astype(jnp.float32)
y = logpk_dz.astype(jnp.float32)

# --- Train/val split ---
N = logpk.shape[0]
split_idx = int(0.9 * N)
X_P_train, X_P_val = X_P[:split_idx], X_P[split_idx:]
X_H_train, X_H_val = X_H[:split_idx], X_H[split_idx:]
X_rho_train, X_rho_val = X_rho[:split_idx], X_rho[split_idx:]
X_z_train, X_z_val = X_z[:split_idx], X_z[split_idx:]
y_train, y_val = y[:split_idx], y[split_idx:]

# --- Neural network model ---
class RHS(eqx.Module):
    mlp: eqx.nn.MLP
    def __init__(self, key):
        self.mlp = eqx.nn.MLP(in_size=265, out_size=262, width_size=512, depth=4, key=key)

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
    return jnp.mean(losses)  # ← Return scalar only

@eqx.filter_jit
def step(model_params, model, opt_state, X_P, X_H, X_rho, X_z, y):
    loss, grads = loss_fn(model_params, model, X_P, X_H, X_rho, X_z, y)
    updates, opt_state = opt.update(grads, opt_state)
    model_params = optax.apply_updates(model_params, updates)
    return model_params, opt_state, loss

# --- Training loop ---
save_path = "/srv/scratch3/taylor.4264/odd_emu/models_final"
os.makedirs(save_path, exist_ok=True)

for run_idx in range(1):
    key = jax.random.PRNGKey(run_idx + 5)
    model = RHS(key)
    model_params = eqx.filter(model, eqx.is_inexact_array)
    opt = optax.adam(1e-3)
    opt_state = opt.init(model_params)

    best_val_loss = jnp.inf
    best_model_params = None
    patience = 20
    wait = 0
    max_epochs = 1000
    batch_size = 32
    num_batches = split_idx // batch_size
    rng = jax.random.PRNGKey(run_idx + 1000)

    for epoch in range(max_epochs):
        rng, subkey = jax.random.split(rng)
        perm = jax.random.permutation(subkey, split_idx)
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
            model_params, opt_state, batch_loss = step(model_params, model, opt_state, X_P_batch, X_H_batch, X_rho_batch, X_z_batch, y_batch)
            epoch_loss += batch_loss

        epoch_loss /= num_batches
        val_loss, _ = loss_fn(model_params, model, X_P_val, X_H_val, X_rho_val, X_z_val, y_val)

        if epoch % 10 == 0:
            print(f"Run {run_idx}, Epoch {epoch}: Train Loss = {epoch_loss:.6e}, Val Loss = {val_loss:.6e}")

        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_model_params = model_params
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"Early stopping run {run_idx} at epoch {epoch}. Best Val Loss = {best_val_loss:.6e}")
                break

    model_file = os.path.join(save_path, f"learned_model_logpk_{run_idx}.eqx")
    eqx.tree_serialise_leaves(model_file, best_model_params)
    print(f"Run {run_idx}: Saved best model to {model_file}")