"""Spec for the physical LEDArray -> FPM bridge (led_array.fpm_shifts / fpm_setup).

Proves the EXISTING fpm.py solver reconstructs when driven by real array geometry
instead of the abstract led_grid_shifts, and that dark-field LEDs deliver
resolution beyond the objective (the point of FPM). fpm.py itself is untouched.
"""

import jax
import jax.numpy as jnp

from bsccm_jax import fpm, led_array

N = 32
NA = 0.25
WL = 0.525
PX = 0.33
UP = 2
ARR = led_array.LEDArray(n=13, pitch_mm=1.5, height_mm=9.0)  # odd -> a centre LED


def _corr(a, b):
    a = jnp.abs(a); b = jnp.abs(b)      # compare magnitudes (global phase is free)
    a = a - a.mean(); b = b - b.mean()
    return float((a * b).sum() / (jnp.linalg.norm(a) * jnp.linalg.norm(b)))


def _amplitude_phantom(hr, cutoff_px):
    """A smooth amplitude object band-limited to the synthetic aperture.

    Amplitude-dominated (weak phase) so it is a clean test of the bridge + FPM's
    resolution gain rather than of phase-retrieval nonconvexity; band-limited to
    the synthetic NA so it is representable, with content past the objective pupil
    so a single brightfield image cannot capture it.
    """
    yy, xx = jnp.mgrid[0:hr[0], 0:hr[1]] - jnp.array([hr[0] // 2, hr[1] // 2])[:, None, None]
    r2 = xx ** 2 + yy ** 2
    spec = (jax.random.normal(jax.random.PRNGKey(1), hr)
            * jnp.exp(-r2 / (2 * (cutoff_px / 2) ** 2)) * (r2 <= cutoff_px ** 2))
    a = fpm.iF(spec).real
    a = 1.0 + 0.5 * (a - a.min()) / (a.max() - a.min() + 1e-9)
    return a.astype(jnp.complex128)


def test_shifts_in_bounds_with_darkfield():
    shifts = led_array.fpm_shifts(ARR, NA, WL, PX, (N, N), UP)
    r = led_array.pupil_radius_px(NA, WL, PX, (N, N))
    lim = (N * UP) // 2 - N // 2
    assert shifts.shape[1] == 2 and shifts.shape[0] > 20
    assert int(jnp.abs(shifts).max()) <= lim                    # crops stay in-bounds
    assert int((jnp.sum(shifts ** 2, axis=1) == 0).sum()) >= 1  # an on-axis LED exists
    maxr = float(jnp.sqrt(jnp.max(jnp.sum(shifts ** 2, axis=1))))
    assert maxr > r                                             # dark-field beyond pupil


def test_fpm_setup_shapes_and_pupil():
    shifts, pupil, hr = led_array.fpm_setup(ARR, (N, N), na=NA, pixel_size_um=PX,
                                            wavelength_um=WL, upsample=UP)
    assert pupil.shape == (N, N)
    assert hr == (N * UP, N * UP)
    r = led_array.pupil_radius_px(NA, WL, PX, (N, N))
    assert abs(float(pupil.sum()) - 3.14159 * r * r) / (3.14159 * r * r) < 0.25


def test_existing_fpm_reconstructs_through_bridge():
    shifts, pupil, hr = led_array.fpm_setup(ARR, (N, N), na=NA, pixel_size_um=PX,
                                            wavelength_um=WL, upsample=UP)
    r_pup = led_array.pupil_radius_px(NA, WL, PX, (N, N))
    r_max = float(jnp.sqrt(jnp.max(jnp.sum(shifts ** 2, axis=1))))
    obj = _amplitude_phantom(hr, r_pup + r_max)                 # within synthetic NA

    imgs = fpm.fpm_forward(obj, shifts, pupil, (N, N))
    assert imgs.shape == (shifts.shape[0], N, N) and jnp.isfinite(imgs).all()

    rec = fpm.reconstruct_fpm(imgs, shifts, pupil, hr, steps=800)

    bf = int(jnp.argmin(jnp.sum(shifts ** 2, axis=1)))          # most on-axis LED
    bf_img = jax.image.resize(jnp.sqrt(imgs[bf]), hr, "bilinear")
    c_rec = _corr(rec, obj)
    c_bf = _corr(bf_img, obj)
    assert c_rec > 0.9              # recovers the high-res object through the bridge
    assert c_rec > c_bf + 0.2       # FPM resolves well beyond a single brightfield image


def test_fpm_reconstructs_under_sensor_noise():
    """A step toward real data: FPM through the bridge survives shot + read noise."""
    shifts, pupil, hr = led_array.fpm_setup(ARR, (N, N), na=NA, pixel_size_um=PX,
                                            wavelength_um=WL, upsample=UP)
    r_pup = led_array.pupil_radius_px(NA, WL, PX, (N, N))
    r_max = float(jnp.sqrt(jnp.max(jnp.sum(shifts ** 2, axis=1))))
    obj = _amplitude_phantom(hr, r_pup + r_max)
    imgs = fpm.fpm_forward(obj, shifts, pupil, (N, N))

    # scale to a photon budget, add Poisson(shot)+Gaussian(read) noise, rescale
    photons = 400.0
    scale = photons / (imgs.mean() + 1e-9)
    k1, k2 = jax.random.split(jax.random.PRNGKey(0))
    counts = imgs * scale
    shot = jnp.sqrt(jnp.clip(counts, 0.0)) * jax.random.normal(k1, imgs.shape)
    read = 5.0 * jax.random.normal(k2, imgs.shape)
    noisy = jnp.clip(counts + shot + read, 0.0) / scale

    rec = fpm.reconstruct_fpm(noisy, shifts, pupil, hr, steps=800)
    assert _corr(rec, obj) > 0.8   # degrades from ~0.97 clean but still recovers
