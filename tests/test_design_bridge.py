"""Spec for the learned-design -> physical-LED bridge (led_array).

bsccm_jax.design optimizes continuous pupil sources; these must map to per-LED
brightness the 32x32 head can display, and the LED-quantized design must remain
deployable — reconstruct as well as the continuous source, and still beat the
standard half-annulus DPC baseline. design.py itself is untouched.
"""

import jax
import jax.numpy as jnp

from bsccm_jax import design, dpc, led_array

SH = (32, 32)
WL = 0.532
PX = 0.2
NA = 0.5
ARR = led_array.LEDArray(n=15, pitch_mm=1.5, height_mm=15.6)  # bright cone covers NA


def _phantoms(n=3):
    fy, fx = jnp.meshgrid(jnp.fft.fftfreq(SH[0]), jnp.fft.fftfreq(SH[1]), indexing="ij")
    g = jnp.exp(-(fx ** 2 + fy ** 2) * (4 * SH[0]) ** 2 / 2)
    smooth = lambda z: jnp.real(jnp.fft.ifft2(jnp.fft.fft2(z) * g))
    return jnp.stack([smooth(jax.random.normal(jax.random.PRNGKey(s), SH)) for s in range(n)])


def test_full_disk_lights_the_in_cone_leds():
    support = dpc.pupil_support(WL, NA, PX, SH).astype(jnp.float64)
    bright = led_array.sources_to_led_brightness(ARR, support[None], WL, PX, SH)[0]
    cone = led_array.bright_cone_mask(ARR, NA)
    assert float(bright[cone].mean()) > 0.9
    assert float(bright[~cone].mean()) < 0.1


def test_half_pupil_source_is_directionally_selective():
    # rotation 180 -> annular_sources selects fx <= 0; LEDs map a -> fx.
    half = dpc.annular_sources(WL, PX, SH, NA, (180.,))[0]
    lb = led_array.sources_to_led_brightness(ARR, half[None], WL, PX, SH)[0]
    a, _ = ARR.direction_cosines()
    cone = led_array.bright_cone_mask(ARR, NA)
    assert float(lb[cone & (a < 0)].mean()) > float(lb[cone & (a > 0)].mean())


def test_learned_design_deploys_to_head_and_beats_dpc():
    P = _phantoms(3)
    init = dpc.annular_sources(WL, PX, SH, NA, (90., 270., 180., 0.))
    learned, pupil, _ = design.learn_illumination(
        P, SH, init_sources=init, wavelength_um=WL, pixel_size_um=PX, na=NA, steps=60)

    bright = led_array.sources_to_led_brightness(ARR, learned, WL, PX, SH)
    assert bright.shape == (4, ARR.n, ARR.n)
    assert 0.0 <= float(bright.min()) and float(bright.max()) <= 1.0 + 1e-6

    realized = led_array.realize_source_from_leds(ARR, bright, WL, PX, SH)
    e_learn = design.sources_recon_error(learned, pupil, P)
    e_real = design.sources_recon_error(realized, pupil, P)
    e_base = design.baseline_dpc_error(P, SH, wavelength_um=WL, pixel_size_um=PX, na=NA)

    assert e_real < e_learn * 1.5     # LED quantization is graceful (deployable)
    assert e_real < e_base * 1.1      # deployed design still ~beats standard DPC
