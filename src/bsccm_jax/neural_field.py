"""Neural-field DPC reconstruction — NSTM's approach in the Kidger stack.

Reimplements the core idea of NSTM (Cao et al., "Neural space-time model for
dynamic multi-shot imaging," Nature Methods 2024; github.com/rmcao/nstm) using
Equinox + Optax instead of Flax + the `cc`/calcil library, composed with our
existing DPC WOTF forward model (bsccm_jax.dpc).

The object (phase, optionally absorption) is represented not as a pixel grid but
as a *coordinate network*: phase(y, x) = MLP(γ(y, x)), where γ is an annealed
positional encoding that introduces high frequencies coarse-to-fine over the
optimization (the BARF/NSTM trick — stabilizes the fit and acts as an implicit
smoothness prior). Fitting is gradient descent (Optax Adam) through the
differentiable WOTF forward model.

NSTM's motion MLP (for dynamic/moving samples) is the natural extension; BSCCM's
white blood cells are effectively static per acquisition, so this is the
space-only object field.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import optax

from bsccm_jax import dpc


def dense_coords(shape):
    ys = jnp.linspace(-1.0, 1.0, shape[0])
    xs = jnp.linspace(-1.0, 1.0, shape[1])
    yy, xx = jnp.meshgrid(ys, xs, indexing="ij")
    return jnp.stack([yy.ravel(), xx.ravel()], axis=-1)  # (N, 2)


def annealed_posenc(coords, num_freqs, alpha):
    """Positional encoding with coarse-to-fine Hann annealing (NSTM/BARF style).

    alpha ∈ [0, num_freqs] gates the frequency bands: only bands below alpha are
    fully active, so the fit starts smooth and sharpens as alpha grows.
    """
    bands = jnp.arange(num_freqs)
    freqs = 2.0 ** bands
    xb = jnp.pi * coords[..., None, :] * freqs[:, None]           # (N, F, 2)
    sincos = jnp.concatenate([jnp.sin(xb), jnp.cos(xb)], axis=-1)  # (N, F, 4)
    coef = jnp.clip(alpha - bands, 0.0, 1.0)
    window = 0.5 * (1.0 - jnp.cos(jnp.pi * coef))                  # (F,)
    feat = (sincos * window[None, :, None]).reshape(coords.shape[0], -1)
    return jnp.concatenate([coords, feat], axis=-1)               # (N, 2 + 4F)


class ObjectField(eqx.Module):
    """Coordinate MLP: encoded (y, x) -> (absorption, phase)."""

    mlp: eqx.nn.MLP
    num_freqs: int = eqx.field(static=True)

    def __init__(self, num_freqs, width, depth, *, key):
        self.num_freqs = num_freqs
        self.mlp = eqx.nn.MLP(
            in_size=2 + 4 * num_freqs, out_size=2, width_size=width, depth=depth,
            activation=jax.nn.gelu, key=key,
        )

    def __call__(self, coords, alpha):
        feat = annealed_posenc(coords, self.num_freqs, alpha)
        return jax.vmap(self.mlp)(feat)  # (N, 2)


def reconstruct_neural_field(imgs, *, wavelength_um=0.532, pixel_size_um=0.2, na=0.5,
                             num_freqs=6, width=128, depth=4, steps=800, lr=2e-3,
                             phase_only=True, seed=0, return_history=False):
    """Fit an Equinox neural field to real DPC images through the WOTF model.

    Returns the reconstructed phase (H, W); optionally the loss history.
    """
    shape = jnp.asarray(imgs["Top"]).shape
    meas = dpc.dpc_measurements(imgs)
    Hp = dpc.dpc_2axis_transfer(shape, wavelength_um=wavelength_um,
                                pixel_size_um=pixel_size_um, na=na)
    coords = dense_coords(shape)
    model = ObjectField(num_freqs, width, depth, key=jax.random.PRNGKey(seed))

    def loss_fn(model, alpha):
        up = model(coords, alpha)
        phase = up[:, 1].reshape(shape)
        pred = dpc.dpc_apply_phase(Hp, phase)
        return jnp.mean((pred - meas) ** 2)

    opt = optax.adam(lr)
    opt_state = opt.init(eqx.filter(model, eqx.is_array))

    @eqx.filter_jit
    def step(model, opt_state, alpha):
        loss, grads = eqx.filter_value_and_grad(loss_fn)(model, alpha)
        updates, opt_state = opt.update(grads, opt_state)
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss

    history = []
    for i in range(steps):
        # coarse-to-fine anneal 0 -> num_freqs; pass as a traced array so the
        # jitted step compiles ONCE (a python float would recompile every step).
        alpha = jnp.asarray(num_freqs * (i + 1) / steps)
        model, opt_state, loss = step(model, opt_state, alpha)
        if return_history and (i % 50 == 0 or i == steps - 1):
            history.append(float(loss))

    phase = model(coords, float(num_freqs))[:, 1].reshape(shape)
    return (phase, history) if return_history else phase
