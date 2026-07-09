"""Fourier Ptychographic Microscopy (FPM) reconstruction — Tier 2.

FPM recovers a high-resolution complex object (amplitude + phase) from a stack of
LOW-resolution intensity images taken under angled LED illumination. Each
illumination angle shifts the object's spectrum under the objective's finite-NA
pupil, so different tilts sample different high-frequency regions; stitching them
recovers a *synthetic aperture* wider than the objective — resolution beyond the
NA limit. This is the reconstruction target for BSCCM's `coherent` (single-LED)
variant.

Built the platform's way: a differentiable JAX forward model + gradient recovery
(Optax), i.e. physics-based learning (Kellman/Lustig/Waller) — autodiff through
the forward, optimize the complex object to match measured intensities. A
classical EPRY alternating-projection solver is also provided as a reference.

Units are pixels of the high-res spectrum. `shifts` are per-LED integer (ky, kx)
offsets of the pupil within that spectrum.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import optax

jax.config.update("jax_enable_x64", True)

F = lambda x: jnp.fft.fftshift(jnp.fft.fft2(jnp.fft.ifftshift(x)))
iF = lambda x: jnp.fft.fftshift(jnp.fft.ifft2(jnp.fft.ifftshift(x)))


def circ_pupil(lr_shape, radius):
    h, w = lr_shape
    yy, xx = jnp.mgrid[0:h, 0:w] - jnp.array([h // 2, w // 2])[:, None, None]
    return (yy ** 2 + xx ** 2 <= radius ** 2).astype(jnp.float64)


def led_grid_shifts(n_side, spacing):
    """(n_side^2, 2) integer (ky,kx) pupil offsets on a centered square LED grid."""
    c = (n_side - 1) / 2
    g = [((i - c) * spacing, (j - c) * spacing) for i in range(n_side) for j in range(n_side)]
    return np.round(np.asarray(g)).astype(int)


def fpm_forward(obj, shifts, pupil, lr_shape):
    """Complex HR object -> (n_led, h, w) low-res intensity images."""
    H, W = obj.shape
    h, w = lr_shape
    O = F(obj)                                   # centered HR spectrum
    cy, cx = H // 2, W // 2

    def one(shift):
        y0 = cy + shift[0] - h // 2
        x0 = cx + shift[1] - w // 2
        sub = jax.lax.dynamic_slice(O, (y0, x0), (h, w))
        field = iF(sub * pupil)                  # low-res coherent field
        return jnp.abs(field) ** 2

    return jax.vmap(one)(jnp.asarray(shifts))


def hr_phantom(hr_shape=(256, 256), seed=0):
    """A high-res complex object: structured phase + mild amplitude."""
    key = jax.random.PRNGKey(seed)
    yy, xx = jnp.mgrid[0:hr_shape[0], 0:hr_shape[1]] / jnp.asarray(hr_shape)[:, None, None]
    ph = jnp.zeros(hr_shape)
    ks = jax.random.split(key, 8)
    for k in ks:                                 # fine features (need super-res)
        c = jax.random.uniform(k, (2,), minval=0.2, maxval=0.8)
        s = jax.random.uniform(k, (), minval=0.01, maxval=0.03)
        ph = ph + jnp.exp(-(((xx - c[1]) ** 2 + (yy - c[0]) ** 2) / (2 * s ** 2)))
    ph = 1.5 * ph / ph.max()
    amp = 1.0 - 0.2 * ph / (ph.max() + 1e-6)
    return amp * jnp.exp(1j * ph)


def reconstruct_fpm(imgs, shifts, pupil, hr_shape, steps=400, lr=5e-2, seed=0):
    """Gradient-based FPM: optimize the complex HR object via Optax Adam.

    The object is parameterized as a real (re, im) pytree — Adam is undefined on
    raw complex arrays (it would square the complex gradient, not its magnitude).
    """
    lr_shape = imgs.shape[1:]
    amp = jax.image.resize(jnp.sqrt(jnp.mean(imgs, axis=0)), hr_shape, "bilinear")
    params = {"re": jnp.asarray(amp, jnp.float64), "im": jnp.zeros(hr_shape)}

    def loss_fn(params):
        obj = params["re"] + 1j * params["im"]
        pred = fpm_forward(obj, shifts, pupil, lr_shape)
        return jnp.mean((jnp.sqrt(pred + 1e-8) - jnp.sqrt(imgs + 1e-8)) ** 2)

    opt = optax.adam(lr)
    state = opt.init(params)

    @jax.jit
    def step(params, state):
        loss, g = jax.value_and_grad(loss_fn)(params)
        updates, state = opt.update(g, state)
        return optax.apply_updates(params, updates), state, loss

    for _ in range(steps):
        params, state, _ = step(params, state)
    return params["re"] + 1j * params["im"]


def reconstruct_fpm_epry(imgs, shifts, pupil, hr_shape, iters=12, update_pupil=True,
                         alpha=1.0, beta=1.0, correct_positions=True, search=1, warmup=2):
    """EPRY alternating-projection FPM — robust on real data (NumPy).

    Embedded Pupil-function Recovery for FPM (Ou et al. 2014): sweep LEDs, replace
    each low-res field's amplitude with the measured sqrt(intensity), and update
    the object spectrum (and pupil) in that region. Initialized from the
    brightfield (most on-axis LED) upsampled by Fourier zero-padding.

    correct_positions: per-LED position correction — after `warmup` iters, each
    LED's pupil shift is refined by a local ±`search`-pixel search that minimizes
    the amplitude mismatch, removing the residual-position lattice artifact that
    survives global geometric calibration. Returns (object, pupil, refined_shifts).
    """
    import numpy as np
    from numpy.fft import fft2, ifft2, fftshift, ifftshift
    Fn = lambda x: fftshift(fft2(ifftshift(x)))
    iFn = lambda x: fftshift(ifft2(ifftshift(x)))

    imgs = np.asarray(imgs, np.float64)
    shifts = np.asarray(shifts, int).copy()
    P = np.asarray(pupil, np.complex128)
    amps = np.sqrt(np.clip(imgs, 0, None))
    h, w = imgs.shape[1:]; H, W = hr_shape; cy, cx = H // 2, W // 2

    bf = amps[int(np.argmin((shifts ** 2).sum(1)))]
    BF = Fn(bf); O = np.zeros((H, W), np.complex128)
    y0i, x0i = cy - h // 2, cx - w // 2                    # odd-safe (h,w may be odd)
    O[y0i:y0i + h, x0i:x0i + w] = BF
    obj = iFn(O); O = Fn(obj)
    cand = [(dy, dx) for dy in range(-search, search + 1) for dx in range(-search, search + 1)]

    for it in range(iters):
        refine = correct_positions and it >= warmup
        for i in range(len(shifts)):
            sy, sx = shifts[i]
            if refine:                                        # local shift search
                best_err, best = np.inf, (sy, sx)
                for dy, dx in cand:
                    yy, xx = cy + sy + dy - h // 2, cx + sx + dx - w // 2
                    if yy < 0 or xx < 0 or yy + h > H or xx + w > W:
                        continue
                    psi_c = iFn(O[yy:yy + h, xx:xx + w] * P)
                    err = np.sum((np.abs(psi_c) - amps[i]) ** 2)
                    if err < best_err:
                        best_err, best = err, (sy + dy, sx + dx)
                sy, sx = best; shifts[i] = [sy, sx]
            y0, x0 = cy + sy - h // 2, cx + sx - w // 2
            sub = O[y0:y0 + h, x0:x0 + w].copy()
            Psi = sub * P
            psi = iFn(Psi)
            dPsi = Fn(amps[i] * np.exp(1j * np.angle(psi))) - Psi
            O[y0:y0 + h, x0:x0 + w] = sub + alpha * np.conj(P) / (np.abs(P).max() ** 2 + 1e-9) * dPsi
            if update_pupil:
                P = P + beta * np.conj(sub) / (np.abs(sub).max() ** 2 + 1e-9) * dPsi
                P *= (np.asarray(pupil) > 0)
    return iFn(O), P, shifts
