"""DUSK sensor-kinetics deconvolution — red-green TDD.

The claim under test is DUSK's central one: solving the binding ODE and fitting a
prior recovers the true concentration waveform better than assuming instantaneous
equilibrium (which leaves the sensor's kinetic blur in place).
"""

import jax.numpy as jnp
import numpy as np

from bsccm_jax import kinetics as K


def _corr(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return float(np.corrcoef(a, b)[0, 1])


def test_binding_ode_lags_equilibrium():
    """The ODE fluorescence should lag a step in [Ca] — that lag is the whole point."""
    kf, kb = K.SENSORS["jgcamp8s"]
    ts = jnp.linspace(0.0, 3.0, 300)
    c = jnp.where(ts > 1.0, 1000.0, 20.0)               # a step up in calcium
    g = np.asarray(K.binding_forward(c, ts, kf, kb))
    g_eq = np.asarray(K.equilibrium_forward(c, kf, kb))
    i = int(np.argmax(ts > 1.0))
    # right after the step, the kinetic g has not yet caught up to equilibrium
    assert g[i + 2] < g_eq[i + 2]
    # and long after, it converges to equilibrium
    assert abs(g[-1] - g_eq[-1]) < 0.02 * g_eq[-1]


def test_dusk_beats_equilibrium():
    """DUSK recovery of [Ca] correlates with ground truth far better than equilibrium."""
    kf, kb = K.SENSORS["jgcamp8s"]
    ts = jnp.linspace(0.0, 5.0, 400)
    c_true = np.asarray(K.calcium_transients(ts))
    f = K.simulate(c_true, ts, kf, kb, seed=1)          # default (noisy) measurement
    c_eq = K.recover_equilibrium(f, kf, kb)
    c_dusk = K.recover_dusk(f, ts, kf, kb)              # default 2000 steps
    r_eq, r_dusk = _corr(c_true, c_eq), _corr(c_true, c_dusk)
    assert r_dusk > r_eq + 0.15                         # a clear, not marginal, win


def test_recovered_concentration_nonnegative():
    """The deep prior enforces c >= 0 (softplus)."""
    kf, kb = K.SENSORS["jgcamp8s"]
    ts = jnp.linspace(0.0, 4.0, 300)
    f = K.simulate(K.calcium_transients(ts), ts, kf, kb, seed=2)
    c = np.asarray(K.recover_dusk(f, ts, kf, kb, steps=300))
    assert c.min() >= 0.0
