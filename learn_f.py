import jax
import jax.numpy as jnp
import numpy as np
import equinox as eqx
import optax
import os

# --- Load Data ---
parent_dir = '/srv/scratch2/taylor.4264/odd_emu/batched/'
Hz_all = np.load(parent_dir + "Hz_all.npy")         # shape (30000, 50)
pk_all = np.load(parent_dir + "pk_nl_all.npy")      # shape (30000, 50, 262)
z_grid = np.load(parent_dir + "z.npy")              # shape (50,)

# --- Compute Derivatives via Finite Differences ---
dz = jnp.diff(z_grid)                          # shape (49,)
pk_diff = pk_all[:, 1:, :] - pk_all[:, :-1, :] # shape (30000, 49, 262)
dpdz = pk_diff / dz[None, :, None]             # shape (30000, 49, 262)

# --- Prepare Inputs ---
P_input = pk_all[:, :-1, :]                    # shape (30000, 49, 262)
H_input = Hz_all[:, :-1]                       # shape (30000, 49)
z_input = z_grid[:-1]                          # shape (49,)
z_input = jnp.broadcast_to(z_input[None, :], H_input.shape)  # shape (30000, 49)

# --- Flatten for training ---
N = P_input.shape[0] * P_input.shape[1]
X_P = P_input.reshape(N, 262)
X_H = H_input.reshape(N, 1)
X_z = z_input.reshape(N, 1)
y = dpdz.reshape(N, 262)

# --- Train/Val Split ---
split_idx = int(0.9 * N)
X_P_train, X_P_val = X_P[:split_idx], X_P[split_idx:]
X_H_train, X_H_val = X_H[:split_idx], X_H[split_idx:]
X_z_train, X_z_val = X_z[:split_idx], X_z[split_idx:]
y_train, y_val = y[:split_idx], y[split_idx:]

# --- Define Vector-Valued RHS Network ---
class RHS(eqx.Module):
    mlp: eqx.nn.MLP
    def __init__(self, key):
        self.mlp = eqx.nn.MLP(in_size=264, out_size=262, width_size=512, depth=4, key=key)

    def __call__(self, P, H, z):
        x = jnp.concatenate([P, H, z])
        return self.mlp(x)

# --- Initialize model and optimizer ---
key = jax.random.PRNGKey(0)
model = RHS(key)
model_params = eqx.filter(model, eqx.is_inexact_array)
opt = optax.adam(1e-3)
opt_state = opt.init(model_params)

# --- Define loss and step functions ---
@eqx.filter_value_and_grad
def loss_fn(model_params, model, X_P, X_H, X_z, y):
    model = eqx.combine(model_params, model)
    def single_example(p, h, z, y_true):
        y_pred = model(p, h, z)
        return jnp.mean((y_pred - y_true) ** 2)
    losses = jax.vmap(single_example)(X_P, X_H, X_z, y)
    return jnp.mean(losses)

@eqx.filter_jit
def step(model_params, model, opt_state, X_P, X_H, X_z, y):
    loss, grads = loss_fn(model_params, model, X_P, X_H, X_z, y)
    updates, opt_state = opt.update(grads, opt_state)
    model_params = optax.apply_updates(model_params, updates)
    return model_params, opt_state, loss

# --- Early stopping config ---
best_val_loss = jnp.inf
best_model_params = None
patience = 20
wait = 0
max_epochs = 1000

# --- Training Loop with Early Stopping ---
for epoch in range(max_epochs):
    model_params, opt_state, train_loss = step(model_params, model, opt_state, X_P_train, X_H_train, X_z_train, y_train)
    val_loss, _ = loss_fn(model_params, model, X_P_val, X_H_val, X_z_val, y_val)

    if epoch % 10 == 0:
        print(f"Epoch {epoch}: Train Loss = {train_loss:.6e}, Val Loss = {val_loss:.6e}")

    if val_loss < best_val_loss - 1e-6:
        best_val_loss = val_loss
        best_model_params = model_params
        wait = 0
    else:
        wait += 1
        if wait >= patience:
            print(f"Early stopping at epoch {epoch}. Best Val Loss = {best_val_loss:.6e}")
            break

# --- Save best model ---
save_path = "/srv/scratch2/taylor.4264/odd_emu/models"
os.makedirs(save_path, exist_ok=True)
model_file = os.path.join(save_path, "learned_model.eqx")
eqx.tree_serialise_leaves(model_file, best_model_params)
print(f"Best model saved to {model_file}")