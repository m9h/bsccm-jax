"""NeuPh — Local Conditional Neural Field for generalizable phase retrieval.

Reimplements the core idea of NeuPh/LCNF (Wang, Zhu, Li, Yang & Tian, Advanced
Photonics Nexus 3(5):056005, 2024) in the Kidger stack. It is the synthesis of
our two prototypes:

  * a CNN ENCODER maps the measurements to a spatial latent feature map
    (like `amortized.py` — gives generalization across cells, single-pass), and
  * a coordinate-MLP DECODER (a neural field, like `neural_field.py`) renders the
    phase at each output coordinate, CONDITIONED on the local latent sampled
    there (+ annealed positional encoding) — giving a continuous, potentially
    resolution-enhanced object.

Trained across a set of objects, it reconstructs a held-out object in one forward
pass — the property per-cell fitting cannot offer at scale. Here trained
supervised on synthetic (phase -> DPC) pairs for a fast, clear demo; swapping the
loss for the DPC forward-consistency (bsccm_jax.dpc) makes it self-supervised.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import optax

from bsccm_jax.neural_field import annealed_posenc, dense_coords


class Encoder(eqx.Module):
    convs: list

    def __init__(self, in_ch, latent, width, key):
        k = jax.random.split(key, 3)
        self.convs = [
            eqx.nn.Conv2d(in_ch, width, 3, padding=1, key=k[0]),
            eqx.nn.Conv2d(width, width, 3, padding=1, key=k[1]),
            eqx.nn.Conv2d(width, latent, 3, padding=1, key=k[2]),
        ]

    def __call__(self, x):
        x = jax.nn.gelu(self.convs[0](x))
        x = jax.nn.gelu(self.convs[1](x))
        return self.convs[2](x)               # (latent, H, W)


class LCNF(eqx.Module):
    """measurements (C, H, W) -> phase (H, W) via local-conditional neural field."""

    encoder: Encoder
    decoder: eqx.nn.MLP
    num_freqs: int = eqx.field(static=True)

    def __init__(self, in_ch=2, latent=16, enc_width=32, num_freqs=4,
                 dec_width=128, dec_depth=3, *, key):
        ke, kd = jax.random.split(key)
        self.encoder = Encoder(in_ch, latent, enc_width, ke)
        self.num_freqs = num_freqs
        self.decoder = eqx.nn.MLP(
            in_size=latent + 2 + 4 * num_freqs, out_size=1,
            width_size=dec_width, depth=dec_depth, activation=jax.nn.gelu, key=kd)

    def __call__(self, meas, alpha=None):
        C, H, W = meas.shape if meas.ndim == 3 else (1, *meas.shape)
        lat = self.encoder(meas)                                  # (latent, H, W)
        coords = dense_coords((H, W))                             # (H*W, 2)
        a = self.num_freqs if alpha is None else alpha
        pe = annealed_posenc(coords, self.num_freqs, a)           # (H*W, 2+4F)
        lat_flat = lat.reshape(lat.shape[0], -1).T                # (H*W, latent)
        feat = jnp.concatenate([lat_flat, pe], axis=-1)
        return jax.vmap(self.decoder)(feat).reshape(H, W)


def train_lcnf(meas_stack, phase_stack, *, latent=16, num_freqs=4, epochs=60,
               batch=8, lr=1e-3, seed=0, val_frac=0.2):
    """Supervised training across (measurements, phase) pairs. Returns (model, val_corr)."""
    meas_stack = jnp.asarray(meas_stack, jnp.float64)
    phase_stack = jnp.asarray(phase_stack, jnp.float64)
    n = len(meas_stack); nval = max(1, int(n * val_frac))
    tr_m, tr_p = meas_stack[nval:], phase_stack[nval:]
    va_m, va_p = meas_stack[:nval], phase_stack[:nval]

    model = LCNF(in_ch=meas_stack.shape[1], latent=latent, num_freqs=num_freqs,
                 key=jax.random.PRNGKey(seed))
    opt = optax.adam(lr)
    state = opt.init(eqx.filter(model, eqx.is_array))

    def loss_fn(model, mb, pb):
        pred = jax.vmap(lambda m: model(m))(mb)
        return jnp.mean((pred - pb) ** 2)

    @eqx.filter_jit
    def step(model, state, mb, pb):
        loss, g = eqx.filter_value_and_grad(loss_fn)(model, mb, pb)
        upd, state = opt.update(g, state)
        return eqx.apply_updates(model, upd), state, loss

    key = jax.random.PRNGKey(seed + 1)
    steps = max(1, tr_m.shape[0] // batch)
    for _ in range(epochs):
        key, sk = jax.random.split(key)
        perm = jax.random.permutation(sk, tr_m.shape[0])
        for s in range(steps):
            idx = perm[s * batch:(s + 1) * batch]
            model, state, _ = step(model, state, tr_m[idx], tr_p[idx])

    # generalization: single-pass reconstruction on held-out
    def corr(a, b):
        a = a.ravel() - a.mean(); b = b.ravel() - b.mean()
        return (a * b).sum() / (jnp.linalg.norm(a) * jnp.linalg.norm(b) + 1e-12)
    preds = jax.vmap(lambda m: model(m))(va_m)
    val_corr = float(jnp.mean(jax.vmap(corr)(preds, va_p)))
    return model, val_corr


def train_lcnf_selfsupervised(meas_stack, Hp, *, latent=16, num_freqs=4, epochs=80,
                              batch=8, lr=1e-3, seed=0, val_frac=0.2, phase_stack=None):
    """Train the LCNF with NO phase labels — physics forward-consistency only.

    Loss = || DPC_forward(LCNF(meas)) - meas ||^2, i.e. the predicted phase must
    reproduce the measurements through the (differentiable) DPC WOTF operator
    ``Hp`` (from dpc.dpc_2axis_transfer). This is NeuPh/PtychoPINN's real mode:
    generalizable reconstruction learned from measurements alone. ``phase_stack``
    is optional and used only to report held-out correlation, never for training.
    """
    from bsccm_jax.dpc import dpc_apply_phase

    meas_stack = jnp.asarray(meas_stack, jnp.float64)
    n = len(meas_stack); nval = max(1, int(n * val_frac))
    tr_m, va_m = meas_stack[nval:], meas_stack[:nval]

    model = LCNF(in_ch=meas_stack.shape[1], latent=latent, num_freqs=num_freqs,
                 key=jax.random.PRNGKey(seed))
    opt = optax.adam(lr)
    state = opt.init(eqx.filter(model, eqx.is_array))

    def loss_fn(model, mb):                       # self-supervised: no labels
        phase = jax.vmap(lambda m: model(m))(mb)
        pred_meas = jax.vmap(lambda p: dpc_apply_phase(Hp, p))(phase)
        return jnp.mean((pred_meas - mb) ** 2)

    @eqx.filter_jit
    def step(model, state, mb):
        loss, g = eqx.filter_value_and_grad(loss_fn)(model, mb)
        upd, state = opt.update(g, state)
        return eqx.apply_updates(model, upd), state, loss

    key = jax.random.PRNGKey(seed + 1)
    steps = max(1, tr_m.shape[0] // batch)
    for _ in range(epochs):
        key, sk = jax.random.split(key)
        perm = jax.random.permutation(sk, tr_m.shape[0])
        for s in range(steps):
            idx = perm[s * batch:(s + 1) * batch]
            model, state, _ = step(model, state, tr_m[idx])

    val_corr = None
    if phase_stack is not None:
        def corr(a, b):
            a = a.ravel() - a.mean(); b = b.ravel() - b.mean()
            return (a * b).sum() / (jnp.linalg.norm(a) * jnp.linalg.norm(b) + 1e-12)
        va_p = jnp.asarray(phase_stack, jnp.float64)[:nval]
        preds = jax.vmap(lambda m: model(m))(va_m)
        # phase-only forward has a sign/scale gauge; report |corr|
        val_corr = float(jnp.mean(jnp.abs(jax.vmap(corr)(preds, va_p))))
    return model, val_corr
