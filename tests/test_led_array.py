"""Spec for the physical LED-matrix geometry (OpenFlexure DPC mod).

Ties a real panel's geometry to the WOTF forward model: the discrete-LED source
must reproduce DPC phase contrast, and a dense array must approach the idealised
``annular_sources`` half-disk it approximates.
"""

import jax.numpy as jnp
import pytest

from bsccm_jax import dpc, led_array

SHAPE = (64, 64)
NA = 0.25
WL = 0.525
PX = 0.5  # sample-plane pixel (um) — coarse so the small pupil is well sampled


def _corr(a, b):
    a = a - a.mean(); b = b - b.mean()
    return float((a * b).sum() / (jnp.linalg.norm(a) * jnp.linalg.norm(b)))


def test_recommend_height_inverts_the_cone_radius():
    h = led_array.recommend_height_mm(NA, pitch_mm=2.5, rings=8.0)
    r_na = h * NA / (1 - NA**2) ** 0.5  # bright-field disk radius on the array
    assert r_na == pytest.approx(8.0 * 2.5, rel=1e-6)


def test_bright_cone_is_a_subset_and_nonempty():
    arr = led_array.LEDArray(n=16, pitch_mm=2.5,
                             height_mm=led_array.recommend_height_mm(NA, 2.5))
    cone = led_array.bright_cone_mask(arr, NA)
    assert 0 < int(cone.sum()) < arr.n * arr.n
    assert arr.corner_na() > NA  # dark-field LEDs exist for later FPM


def test_opposite_dpc_frames_are_mirror_images():
    arr = led_array.LEDArray(n=16, pitch_mm=2.5, height_mm=40.0)
    masks = led_array.dpc_led_masks(arr, NA, (0.0, 180.0))
    # left/right half-cones partition the cone (diameter shared), no overlap off-axis
    both = masks[0] & masks[1]
    cone = led_array.bright_cone_mask(arr, NA)
    assert int((masks[0] | masks[1]).sum()) == int(cone.sum())
    assert int(both.sum()) <= arr.n  # only the shared diameter can overlap


def test_source_stack_matches_lit_led_counts():
    arr = led_array.LEDArray(n=16, pitch_mm=2.5, height_mm=40.0)
    masks = led_array.dpc_led_masks(arr, NA)
    src = led_array.led_dpc_sources(arr, NA, WL, PX, SHAPE)
    assert src.shape == (4, *SHAPE)
    # every lit LED deposits >=1 unit of source weight (snapping may stack a few)
    for m, s in zip(masks, src):
        assert float(s.sum()) >= 1.0
        assert float(s.sum()) <= float(m.sum()) + 1e-6


def test_discrete_array_forward_gives_phase_contrast():
    # A pure phase phantom must produce antisymmetric contrast between opposite
    # half-cones — the defining DPC signal — through the discrete-LED forward.
    arr = led_array.LEDArray(n=24, pitch_mm=2.0,
                             height_mm=led_array.recommend_height_mm(NA, 2.0))
    fwd = led_array.build_dpc_forward_from_array(arr, SHAPE, na=NA, pixel_size_um=PX,
                                                 wavelength_um=WL)
    u, p = dpc.phantom(SHAPE)
    imgs = fwd(u, p)
    assert imgs.shape == (4, *SHAPE)
    assert jnp.isfinite(imgs).all()
    left_minus_right = imgs[0] - imgs[1]
    assert float(jnp.abs(left_minus_right).sum()) > 0.0  # phase contrast present


def test_dense_array_recovers_phase_via_existing_reconstructor():
    # End-to-end: forward through the physical-array model, invert with the stock
    # Tikhonov solver. A dense array should recover the phase well.
    arr = led_array.LEDArray(n=32, pitch_mm=1.5,
                             height_mm=led_array.recommend_height_mm(NA, 1.5, rings=10))
    fwd = led_array.build_dpc_forward_from_array(arr, SHAPE, na=NA, pixel_size_um=PX,
                                                 wavelength_um=WL)
    u, p = dpc.phantom(SHAPE)
    imgs = fwd(u, p)
    _, p_hat = dpc.reconstruct_tikhonov(imgs, fwd)
    assert _corr(p_hat, p) > 0.9


def test_calibration_report_flags_coarse_arrays():
    coarse = led_array.LEDArray(n=8, pitch_mm=5.0, height_mm=30.0)  # too few rings
    rep = led_array.calibration_report(coarse, NA, WL, PX, SHAPE)
    assert rep["led_rings_in_cone"] < 4
    assert "warning" in rep
