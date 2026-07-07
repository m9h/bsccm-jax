"""Red-green spec for the DPC forward model and reconstructors.

The phantom is forward-simulated through the Equinox WOTF model, so every
reconstructor has a known ground-truth object to recover. Correlation with the
true phase is the acceptance criterion.
"""

import jax.numpy as jnp
import pytest

from bsccm_jax import dpc

SHAPE = (64, 64)  # small enough for a fast suite


def _corr(a, b):
    a = a - a.mean(); b = b - b.mean()
    return float((a * b).sum() / (jnp.linalg.norm(a) * jnp.linalg.norm(b)))


@pytest.fixture(scope="module")
def problem():
    u, p = dpc.phantom(SHAPE)
    fwd = dpc.DPCForward.build(SHAPE)
    images = fwd(u, p)
    return u, p, fwd, images


def test_forward_produces_one_image_per_source(problem):
    _, _, fwd, images = problem
    assert images.shape == (fwd.Hu.shape[0], *SHAPE)
    assert jnp.isfinite(images).all()


def test_tikhonov_recovers_phase(problem):
    u, p, fwd, images = problem
    _, p_hat = dpc.reconstruct_tikhonov(images, fwd)
    assert _corr(p_hat, p) > 0.95


def test_optimistix_recovers_phase(problem):
    u, p, fwd, images = problem
    _, p_hat = dpc.reconstruct_optimistix(images, fwd, max_steps=300)
    assert _corr(p_hat, p) > 0.90


def test_scico_tv_recovers_phase(problem):
    u, p, fwd, images = problem
    _, p_hat = dpc.reconstruct_scico(images, fwd, max_iter=80)
    assert _corr(p_hat, p) > 0.90


def test_lineax_recovers_phase(problem):
    # Works on jax 0.10.2 / lineax 0.1.1. (Broke under jax 0.7.1, which equinox
    # blacklists: 'cannot create weak reference to Flatten object'.)
    u, p, fwd, images = problem
    _, p_hat = dpc.reconstruct_lineax(images, fwd)
    assert _corr(p_hat, p) > 0.90
