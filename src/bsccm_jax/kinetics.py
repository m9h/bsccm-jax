"""DUSK-style sensor-kinetics deconvolution — the platform's differentiable-ODE tier.

Port of the core idea from Pham, Boquet-Pujadas, Mondal, Unser & Barbastathis,
"Deep-prior ODEs augment fluorescence imaging with chemical sensors"
(Nat. Commun. 2024; github.com/ThanhAnPham/DUSK).

Fluorescent calcium sensors (GCaMP) bind Ca2+ with *finite* kinetics, so the
measured fluorescence is a temporally-blurred, saturating function of the true
[Ca]. Assuming instantaneous equilibrium (g = c/(c+Kd)) smears fast transients
and distorts their timing; solving the binding ODE and fitting a prior recovers
the true concentration waveform — the same underlying event regardless of which
(fast/slow) sensor was used.

This is the *temporal* analogue of our spatial physics-based reconstructors
(dpc/fpm/neuph): a differentiable forward model + a prior, on the Kidger stack —
diffrax (ODE solve) + equinox (deep temporal prior) + optax. It is the first use
of diffrax in the platform.

    dg/dt = kf * c(t) * (1 - g) - kb * g        (1:1 binding, bound fraction g)
    equilibrium:  g_eq = c / (c + Kd),   Kd = kb / kf

`kf` [nM^-1 s^-1], `kb` [s^-1] from the DUSK repo (GCAMPparam.py). `c` in nM.
"""

from __future__ import annotations

import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
import optax

jax.config.update("jax_enable_x64", True)

# (kf [nM^-1 s^-1], kb [s^-1]) — jgcamp8s is the DUSK default; 8m/7f are faster.
SENSORS = {
    "jgcamp8s": (8.089243550527982e-04, 3.681079025810),
    "jgcamp8m": (1.6e-03, 9.0),
    "jgcamp7f": (2.3e-03, 14.0),
}


def kd(kf, kb):
    return kb / kf


def binding_forward(c_ts, ts, kf, kb, g0=None):
    """Solve dg/dt = kf*c(t)*(1-g) - kb*g for the bound fraction g(t).

    c_ts are concentrations sampled at ts (linearly interpolated inside the
    solve). Differentiable w.r.t. c_ts, so a prior producing c_ts can be fit by
    gradient descent through the ODE.
    """
    c_ts = jnp.asarray(c_ts, jnp.float64)
    ci = diffrax.LinearInterpolation(ts, c_ts)
    if g0 is None:
        c0 = c_ts[0]
        g0 = kf * c0 / (kf * c0 + kb)                     # start at equilibrium
    term = diffrax.ODETerm(lambda t, g, a: kf * ci.evaluate(t) * (1.0 - g) - kb * g)
    sol = diffrax.diffeqsolve(
        term, diffrax.Tsit5(), t0=ts[0], t1=ts[-1], dt0=ts[1] - ts[0],
        y0=g0, saveat=diffrax.SaveAt(ts=ts), max_steps=100_000,
        stepsize_controller=diffrax.PIDController(rtol=1e-6, atol=1e-8))
    return sol.ys


def equilibrium_forward(c, kf, kb):
    """Instantaneous-equilibrium fluorescence (the naive model DUSK improves on)."""
    c = jnp.asarray(c, jnp.float64)
    return c / (c + kd(kf, kb))


def simulate(c_ts, ts, kf, kb, qe=10.0, bg=0.25, photon=12.5, sigma=0.1, seed=0):
    """Kinetic fluorescence + shot(Poisson)+read(Gaussian) noise, DUSK-style."""
    g = binding_forward(c_ts, ts, kf, kb)
    f = qe * g + bg
    key = jax.random.PRNGKey(seed)
    k1, k2 = jax.random.split(key)
    shot = jax.random.poisson(k1, jnp.clip(f * photon, 0, None)) / photon
    return shot + sigma * jax.random.normal(k2, f.shape)


def recover_equilibrium(f, kf, kb, qe=10.0, bg=0.25):
    """Invert the equilibrium model: c = Kd * g / (1 - g). Temporally distorted."""
    g = jnp.clip((jnp.asarray(f) - bg) / qe, 1e-4, 0.999)
    return kd(kf, kb) * g / (1.0 - g)


class _TimePrior(eqx.Module):
    """Deep temporal prior: t in [0,1] -> softplus -> c(t) (>=0). A coordinate
    MLP, the DUSK 'deep image prior' as an untrained net fit to one recording."""
    layers: list
    scale: jax.Array

    def __init__(self, width=64, depth=3, scale=1e3, *, key):
        keys = jax.random.split(key, depth)
        dims = [1] + [width] * (depth - 1) + [1]
        self.layers = [eqx.nn.Linear(dims[i], dims[i + 1], key=keys[i]) for i in range(depth)]
        self.scale = jnp.asarray(float(scale))

    def __call__(self, t):
        x = jnp.atleast_1d(t)
        for lyr in self.layers[:-1]:
            x = jnp.tanh(lyr(x))
        return self.scale * jax.nn.softplus(self.layers[-1](x))[0]


def recover_dusk(f, ts, kf, kb, qe=10.0, bg=0.25, steps=2000, lr=3e-3,
                 tv=5e-4, width=64, depth=3, seed=0):
    """DUSK: fit a deep temporal prior so the ODE forward matches the measured
    fluorescence. Returns the recovered concentration c(t) at ts.

    Loss = || qe*binding_forward(c) + bg - f ||_1 + tv * TV(c), c = prior(t).
    """
    ts = jnp.asarray(ts, jnp.float64)
    f = jnp.asarray(f, jnp.float64)
    tn = (ts - ts[0]) / (ts[-1] - ts[0])                  # normalized time in [0,1]
    model = _TimePrior(width=width, depth=depth,
                       scale=float(jnp.maximum(recover_equilibrium(f, kf, kb, qe, bg).max(), 1.0)),
                       key=jax.random.PRNGKey(seed))

    def c_of(model):
        return jax.vmap(model)(tn)

    def loss_fn(model):
        c = c_of(model)
        g = binding_forward(c, ts, kf, kb)
        data = jnp.mean(jnp.abs(qe * g + bg - f))
        reg = tv * jnp.mean(jnp.abs(jnp.diff(c)))
        return data + reg

    opt = optax.adam(lr)
    state = opt.init(eqx.filter(model, eqx.is_array))

    @eqx.filter_jit
    def step(model, state):
        loss, grads = eqx.filter_value_and_grad(loss_fn)(model)
        updates, state = opt.update(grads, state, eqx.filter(model, eqx.is_array))
        return eqx.apply_updates(model, updates), state, loss

    for _ in range(steps):
        model, state, _ = step(model, state)
    return c_of(model)


def calcium_transients(ts, spikes=((0.4, 800.0), (0.65, 400.0), (0.85, 1000.0)),
                       tau_rise=0.02, tau_decay=0.15, baseline=20.0):
    """Synthetic [Ca] ground truth: fast-rising, exp-decaying spikes on a baseline."""
    ts = jnp.asarray(ts, jnp.float64)
    T = ts[-1] - ts[0]
    c = jnp.full_like(ts, baseline)
    for frac, amp in spikes:
        t0 = ts[0] + frac * T
        dt = ts - t0
        kernel = jnp.where(dt >= 0, (1 - jnp.exp(-dt / tau_rise)) * jnp.exp(-dt / tau_decay), 0.0)
        c = c + amp * kernel
    return c
