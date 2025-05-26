import jax
import jax.numpy as jnp
import numpy as np
import equinox as eqx
import optax
import os
import multiprocessing as mp

def train_model(run_idx, device_id):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
    jax.devices()  # trigger device assignment

    # --- Load Data ---
    parent_dir = '/srv/scratch2/taylor.4264/odd_emu/batched_low_z/'
    Hz_all = jnp.load(parent_dir + "Hz_all.npy")
    pk_all = jnp.load(parent_dir + "pk_nl_all.npy")
    z_grid = jnp.load(parent_dir + "z.npy")

    # --- Derivatives ---
    dz = jnp.diff(z_grid)
    pk_diff = pk_all[:, 1:, :] - pk_all[:, :-1, :]
    dpdz = pk_diff / dz[None, :, None]

    # --- Inputs ---
    P_input = pk_all[:, :-1, :]
    H_input = Hz_all[:, :-1]
    z_input = z_grid[:-1]
    z_input = np.broadcast_to(z_input[None, :], H_input.shape)

    # --- Flatten ---
    N = P_input.shape[0] * P_input.shape[1]
    X_P = jnp.array(P_input.reshape(N, 262))
    X_H = jnp.array(H_input.reshape(N, 1))
    X_z = jnp.array(z_input.reshape(N, 1))
    y = jnp.array(dpdz.reshape(N, 262))

    # --- Train/Val Split ---
    split_idx = int(0.9 * N)
    X_P_train, X_P_val = X_P[:split_idx], X_P[split_idx:]
    X_H_train, X_H_val = X_H[:split_idx], X_H[split_idx:]
    X_z_train, X_z_val = X_z[:split_idx], X_z[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    # --- Model ---
    class RHS(eqx.Module):
        mlp: eqx.nn.MLP
        def __init__(self, key):
            self.mlp = eqx.nn.MLP(in_size=264, out_size=262, width_size=512, depth=4, key=key)

        def __call__(self, P, H, z):
            x = jnp.concatenate([P, H, z])
            return self.mlp(x)

    key = jax.random.PRNGKey(run_idx)
    model = RHS(key)
    model_params = eqx.filter(model, eqx.is_inexact_array)
    opt = optax.adam(1e-3)
    opt_state = opt.init(model_params)

    @eqx.filter_value_and_grad
    def loss_fn(params, model, X_P, X_H, X_z, y):
        model = eqx.combine(params, model)
        def single_example(p, h, z, y_true):
            y_pred = model(p, h, z)
            return jnp.mean((y_pred - y_true) ** 2)
        losses = jax.vmap(single_example)(X_P, X_H, X_z, y)
        return jnp.mean(losses)

    @eqx.filter_jit
    def step(params, model, opt_state, X_P, X_H, X_z, y):
        loss, grads = loss_fn(params, model, X_P, X_H, X_z, y)
        updates, opt_state = opt.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    # --- Early stopping ---
    best_val_loss = jnp.inf
    best_model_params = None
    patience = 20
    wait = 0
    max_epochs = 1000
    batch_size = 15000
    num_batches = split_idx // batch_size
    rng = jax.random.PRNGKey(42)

    for epoch in range(max_epochs):
        perm = jax.random.permutation(rng, split_idx)
        X_P_train = X_P_train[perm]
        X_H_train = X_H_train[perm]
        X_z_train = X_z_train[perm]
        y_train = y_train[perm]

        epoch_loss = 0.0
        for i in range(num_batches):
            start, end = i * batch_size, (i + 1) * batch_size
            X_P_batch = X_P_train[start:end]
            X_H_batch = X_H_train[start:end]
            X_z_batch = X_z_train[start:end]
            y_batch = y_train[start:end]
            model_params, opt_state, batch_loss = step(model_params, model, opt_state, X_P_batch, X_H_batch, X_z_batch, y_batch)
            epoch_loss += batch_loss

        epoch_loss /= num_batches
        val_loss, _ = loss_fn(model_params, model, X_P_val, X_H_val, X_z_val, y_val)

        if epoch % 10 == 0:
            print(f"[Run {run_idx}, GPU {device_id}] Epoch {epoch}: Train Loss = {epoch_loss:.6e}, Val Loss = {val_loss:.6e}")

        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_model_params = model_params
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"[Run {run_idx}] Early stopping at epoch {epoch}. Best Val Loss = {best_val_loss:.6e}")
                break

    # --- Save model ---
    save_path = "/srv/scratch2/taylor.4264/odd_emu/models"
    os.makedirs(save_path, exist_ok=True)
    model_file = os.path.join(save_path, f"learned_model_low_z_{run_idx}.eqx")
    eqx.tree_serialise_leaves(model_file, best_model_params)
    print(f"[Run {run_idx}] Model saved to {model_file}")


# --- Launch parallel training on 2 GPUs ---
if __name__ == "__main__":
    processes = []
    for run_idx in range(20):
        device_id = run_idx % 2  # Alternate between GPU 0 and 1
        p = mp.Process(target=train_model, args=(run_idx, device_id))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()