"""End-to-end validation of the DPC pipeline on a synthetic phantom.

No dataset needed: forward-simulate DPC images from a known object, then confirm
each reconstructor (analytic / Lineax / Optimistix) recovers the phase.
"""

import jax
import jax.numpy as jnp

from bsccm_jax import dpc


def _corr(a, b):
    a = a - a.mean(); b = b - b.mean()
    return float((a * b).sum() / (jnp.linalg.norm(a) * jnp.linalg.norm(b)))


def main():
    shape = (128, 128)
    u_true, p_true = dpc.phantom(shape)
    fwd = dpc.DPCForward.build(shape)

    images = fwd(u_true, p_true)
    key = jax.random.PRNGKey(1)
    images = images + 1e-3 * jax.random.normal(key, images.shape)  # sensor noise
    print(f"forward model: {images.shape[0]} DPC images {images.shape[1:]}, "
          f"transfer fns Hu/Hp {fwd.Hu.shape}")

    methods = {
        "tikhonov (analytic)": dpc.reconstruct_tikhonov,
        "lineax   (NormalCG)": dpc.reconstruct_lineax,
        "optimistix (BFGS)  ": dpc.reconstruct_optimistix,
    }
    print(f"\n{'method':22s}  phase corr   absorb corr")
    print("-" * 50)
    for name, fn in methods.items():
        u_hat, p_hat = fn(images, fwd)
        print(f"{name:22s}   {_corr(p_hat, p_true):+.4f}      {_corr(u_hat, u_true):+.4f}")
    print("\ncorr -> +1.0 means the recovered field matches the ground-truth object.")


if __name__ == "__main__":
    main()
