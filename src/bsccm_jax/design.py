"""Differentiable coded-illumination design (Tier 4 frontier).

End-to-end optimization of the microscope: because the whole pipeline
— illumination pattern -> WOTF -> noisy measurement -> reconstruction -> error —
is differentiable JAX, we can autodiff the reconstruction error with respect to
the *illumination source patterns* and learn the coded illumination that makes
phase most recoverable under noise. This is the design counterpart to IDEAL
(information-driven design) realized concretely with our own forward model, and
the physics-based-learning idea (Kellman/Lustig/Waller) taken one level up: not
just learning the reconstruction, but learning what to *measure*.

The design is meaningful only under noise — without it, any invertible
illumination reconstructs perfectly. With sensor noise, the learned patterns
trade off signal capture against noise robustness.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import optax

from bsccm_jax import dpc

F, iF = dpc.F, dpc.iF


def _phase_tikhonov(Hp, meas, reg):
    """Phase-only reconstruction from K measurements (differentiable, closed form)."""
    fmeas = jax.vmap(F)(meas.astype(jnp.complex128))
    num = (Hp.conj() * fmeas).sum(0)
    den = (Hp.conj() * Hp).sum(0) + reg
    return iF(num / den).real


def sources_from_params(params, support):
    """Non-negative source patterns confined to the illumination support."""
    return jax.nn.softplus(params) * support


def recon_error(params, support, pupil, phantoms, key, noise=0.02, reg=5e-3):
    """Mean phase reconstruction MSE over phantoms, under sensor noise."""
    sources = sources_from_params(params, support)
    _, Hp = dpc.generate_wotf(sources, pupil)

    def one(p, k):
        meas = jax.vmap(lambda h: iF(h * F(p)).real)(Hp)          # (K,H,W) clean
        meas = meas + noise * jax.random.normal(k, meas.shape)    # sensor noise
        rec = _phase_tikhonov(Hp, meas, reg)
        d = rec - p
        return jnp.mean(d ** 2)

    keys = jax.random.split(key, phantoms.shape[0])
    return jnp.mean(jax.vmap(one)(phantoms, keys))


def sources_recon_error(sources, pupil, phantoms, *, noise=0.02, reg=5e-3, seed=1):
    """Noisy phase-recon MSE for an arbitrary set of raw source patterns (fair eval)."""
    _, Hp = dpc.generate_wotf(jnp.asarray(sources), pupil)
    phantoms = jnp.asarray(phantoms, jnp.float64)

    def one(p, k):
        meas = jax.vmap(lambda h: iF(h * F(p)).real)(Hp)
        meas = meas + noise * jax.random.normal(k, meas.shape)
        return jnp.mean((_phase_tikhonov(Hp, meas, reg) - p) ** 2)

    keys = jax.random.split(jax.random.PRNGKey(seed), phantoms.shape[0])
    return float(jnp.mean(jax.vmap(one)(phantoms, keys)))


def learn_illumination(phantoms, shape, *, init_sources, wavelength_um=0.532,
                       pixel_size_um=0.2, na=0.5, noise=0.02, steps=150, lr=2e-2, seed=0):
    """Refine illumination source patterns (from init) to minimize noisy-recon error."""
    pupil = dpc.generate_pupil(wavelength_um, pixel_size_um, na, shape)
    support = dpc.pupil_support(wavelength_um, na, pixel_size_um, shape).astype(jnp.float64)
    phantoms = jnp.asarray(phantoms, jnp.float64)

    # init params so softplus(params)*support == init_sources (inverse softplus)
    s0 = jnp.clip(jnp.asarray(init_sources, jnp.float64), 1e-4, None)
    params = jnp.log(jnp.expm1(jnp.clip(s0, 1e-4, 30.0)) + 1e-9)
    key = jax.random.PRNGKey(seed)
    opt = optax.adam(lr)
    state = opt.init(params)

    @jax.jit
    def step(params, state, k):
        loss, g = jax.value_and_grad(recon_error)(params, support, pupil, phantoms, k, noise)
        upd, state = opt.update(g, state)
        return optax.apply_updates(params, upd), state, loss

    for i in range(steps):
        key, k = jax.random.split(key)
        params, state, _ = step(params, state, k)
    return sources_from_params(params, support), pupil, support


def baseline_dpc_error(phantoms, shape, *, wavelength_um=0.532, pixel_size_um=0.2,
                       na=0.5, noise=0.02, seed=1):
    """Recon error of the standard half-annulus DPC (Top/Bottom/Left/Right)."""
    pupil = dpc.generate_pupil(wavelength_um, pixel_size_um, na, shape)
    src = dpc.annular_sources(wavelength_um, pixel_size_um, shape, na, (90., 270., 180., 0.))
    # differential DPC transfer built the standard way
    _, Hp_each = dpc.generate_wotf(src, pupil)
    Hp = jnp.stack([Hp_each[0] - Hp_each[1], Hp_each[2] - Hp_each[3]])
    phantoms = jnp.asarray(phantoms, jnp.float64)
    key = jax.random.PRNGKey(seed)

    def one(p, k):
        meas = jax.vmap(lambda h: iF(h * F(p)).real)(Hp)
        meas = meas + noise * jax.random.normal(k, meas.shape)
        rec = _phase_tikhonov(Hp, meas, 5e-3)
        return jnp.mean((rec - p) ** 2)

    keys = jax.random.split(key, phantoms.shape[0])
    return float(jnp.mean(jax.vmap(one)(phantoms, keys)))
