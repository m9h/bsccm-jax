"""Amortized, self-supervised, physics-informed DPC reconstruction.

The PtychoPINN paradigm (AdvancedPhotonSource/PtychoPINN-torch-pub) in the Kidger
stack: instead of optimizing each cell independently (as dpc.py / neural_field.py
do), train ONE Equinox conv encoder–decoder that maps DPC measurements -> phase in
a single forward pass. Training is self-supervised — the loss pushes the predicted
phase through our differentiable WOTF forward model and matches the measured
differential images, so no ground-truth phase is ever needed:

    L(θ) = Σ_cells || A( f_θ(meas) ) − meas ||²      (A = DPC forward model)

Once trained, inference on a new cell is a single forward pass (amortized), and the
network generalizes across cells — the property per-cell solvers cannot offer at
BSCCM's 400k-cell scale.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import optax

from bsccm_jax import dpc


class ConvBlock(eqx.Module):
    c1: eqx.nn.Conv2d
    c2: eqx.nn.Conv2d

    def __init__(self, cin, cout, key):
        k1, k2 = jax.random.split(key)
        self.c1 = eqx.nn.Conv2d(cin, cout, 3, padding=1, key=k1)
        self.c2 = eqx.nn.Conv2d(cout, cout, 3, padding=1, key=k2)

    def __call__(self, x):
        return jax.nn.gelu(self.c2(jax.nn.gelu(self.c1(x))))


def _down(x):  # mean-pool /2
    c, h, w = x.shape
    return x.reshape(c, h // 2, 2, w // 2, 2).mean((2, 4))


def _up(x):  # nearest x2
    c, h, w = x.shape
    return jax.image.resize(x, (c, h * 2, w * 2), method="nearest")


class UNet(eqx.Module):
    """Compact 2-level U-Net: (in_ch, H, W) -> (1, H, W) phase."""

    e1: ConvBlock
    e2: ConvBlock
    bott: ConvBlock
    d2: ConvBlock
    d1: ConvBlock
    head: eqx.nn.Conv2d

    def __init__(self, in_ch=2, base=16, *, key):
        k = jax.random.split(key, 6)
        self.e1 = ConvBlock(in_ch, base, k[0])
        self.e2 = ConvBlock(base, base * 2, k[1])
        self.bott = ConvBlock(base * 2, base * 4, k[2])
        self.d2 = ConvBlock(base * 4 + base * 2, base * 2, k[3])
        self.d1 = ConvBlock(base * 2 + base, base, k[4])
        self.head = eqx.nn.Conv2d(base, 1, 1, key=k[5])

    def __call__(self, x):
        s1 = self.e1(x)
        s2 = self.e2(_down(s1))
        b = self.bott(_down(s2))
        d2 = self.d2(jnp.concatenate([_up(b), s2], axis=0))
        d1 = self.d1(jnp.concatenate([_up(d2), s1], axis=0))
        return self.head(d1)[0]  # (H, W)


def train_amortized(meas_stack, *, wavelength_um=0.532, pixel_size_um=0.2, na=0.5,
                    base=16, epochs=40, batch=16, lr=1e-3, seed=0, val_frac=0.2):
    """Self-supervised training over a stack of DPC measurements.

    meas_stack: (N, 2, H, W) normalized differential measurements (from
    dpc.dpc_measurements per cell). Returns (trained_model, history).
    """
    meas_stack = jnp.asarray(meas_stack, jnp.float64)
    n, _, h, w = meas_stack.shape
    shape = (h, w)
    Hp = dpc.dpc_2axis_transfer(shape, wavelength_um=wavelength_um,
                                pixel_size_um=pixel_size_um, na=na)

    n_val = max(1, int(n * val_frac))
    tr, va = meas_stack[n_val:], meas_stack[:n_val]

    model = UNet(in_ch=2, base=base, key=jax.random.PRNGKey(seed))
    opt = optax.adam(lr)
    opt_state = opt.init(eqx.filter(model, eqx.is_array))

    def cell_loss(model, m):                      # m: (2, H, W)
        phase = model(m)                          # (H, W)
        pred = dpc.dpc_apply_phase(Hp, phase)     # (2, H, W)
        return jnp.mean((pred - m) ** 2)

    def batch_loss(model, mb):
        return jnp.mean(jax.vmap(lambda m: cell_loss(model, m))(mb))

    @eqx.filter_jit
    def step(model, opt_state, mb):
        loss, grads = eqx.filter_value_and_grad(batch_loss)(model, mb)
        updates, opt_state = opt.update(grads, opt_state)
        return eqx.apply_updates(model, updates), opt_state, loss

    key = jax.random.PRNGKey(seed + 1)
    history = []
    steps_per_epoch = max(1, tr.shape[0] // batch)
    for ep in range(epochs):
        key, sk = jax.random.split(key)
        perm = jax.random.permutation(sk, tr.shape[0])
        for s in range(steps_per_epoch):
            idx = perm[s * batch:(s + 1) * batch]
            model, opt_state, _ = step(model, opt_state, tr[idx])
        val = float(batch_loss(model, va))
        history.append(val)
    return model, history


@eqx.filter_jit
def infer(model, meas):
    """Amortized phase reconstruction for one cell — a single forward pass."""
    return model(jnp.asarray(meas, jnp.float64))
