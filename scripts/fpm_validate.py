"""Validate FPM through the PHYSICAL LED-array bridge (led_array -> fpm).

Runs the full pipeline your OpenFlexure head will drive: real array geometry ->
shifts -> fpm.reconstruct_fpm, and writes a brightfield-vs-FPM comparison PNG.

    # no rig needed — realistic noisy synthetic through the real geometry:
    python scripts/fpm_validate.py

    # real captures from the head:
    python scripts/fpm_validate.py --captures DIR

`DIR` holds one raw image per LED as `led_000.npy` ... (float, background-subtracted)
plus `geometry.json`:
    {"n":32,"pitch_mm":2.5,"height_mm":78,"na":0.25,"wavelength_um":0.525,
     "pixel_size_um":0.34,"upsample":2}
LED order must match led_array.fpm_shifts (row-major over the n x n array, with
the same out-of-bounds LEDs dropped) — capture with that ordering, or pass an
explicit index map later. Orientation/flip is calibrated per rig (Angle_SelfCalibration).
"""

from __future__ import annotations

import argparse
import json
import pathlib

import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")               # headless: NVIDIA+Xvfb GL is blank, use Agg
import matplotlib.pyplot as plt

from bsccm_jax import fpm, led_array


def _corr(a, b):
    a = jnp.abs(a) - jnp.abs(a).mean()
    b = jnp.abs(b) - jnp.abs(b).mean()
    return float((a * b).sum() / (jnp.linalg.norm(a) * jnp.linalg.norm(b) + 1e-12))


def load_captures(d: pathlib.Path):
    g = json.loads((d / "geometry.json").read_text())
    files = sorted(d.glob("led_*.npy"))
    imgs = jnp.stack([jnp.asarray(jnp.load(f), jnp.float64) for f in files])
    arr = led_array.LEDArray(g["n"], g["pitch_mm"], g["height_mm"])
    shifts, pupil, hr = led_array.fpm_setup(
        arr, imgs.shape[1:], na=g["na"], pixel_size_um=g["pixel_size_um"],
        wavelength_um=g["wavelength_um"], upsample=g.get("upsample", 2))
    assert imgs.shape[0] == shifts.shape[0], (
        f"{imgs.shape[0]} images vs {shifts.shape[0]} in-bounds LEDs — check LED ordering")
    return imgs, shifts, pupil, hr, None


def make_synthetic(photons=400.0):
    n, na, wl, px, up = 32, 0.25, 0.525, 0.33, 2
    arr = led_array.LEDArray(n=13, pitch_mm=1.5, height_mm=9.0)
    shifts, pupil, hr = led_array.fpm_setup(arr, (n, n), na=na, pixel_size_um=px,
                                            wavelength_um=wl, upsample=up)
    r = led_array.pupil_radius_px(na, wl, px, (n, n)) + float(
        jnp.sqrt(jnp.max(jnp.sum(shifts ** 2, axis=1))))
    yy, xx = jnp.mgrid[0:hr[0], 0:hr[1]] - jnp.array([hr[0] // 2, hr[1] // 2])[:, None, None]
    spec = (jax.random.normal(jax.random.PRNGKey(1), hr)
            * jnp.exp(-(xx ** 2 + yy ** 2) / (2 * (r / 2) ** 2)) * (xx ** 2 + yy ** 2 <= r ** 2))
    a = fpm.iF(spec).real
    obj = (1.0 + 0.5 * (a - a.min()) / (a.max() - a.min() + 1e-9)).astype(jnp.complex128)
    imgs = fpm.fpm_forward(obj, shifts, pupil, (n, n))
    scale = photons / (imgs.mean() + 1e-9)
    k1, k2 = jax.random.split(jax.random.PRNGKey(0))
    counts = imgs * scale
    noisy = jnp.clip(counts + jnp.sqrt(jnp.clip(counts, 0.0)) * jax.random.normal(k1, imgs.shape)
                     + 5.0 * jax.random.normal(k2, imgs.shape), 0.0) / scale
    return noisy, shifts, pupil, hr, obj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--captures", type=pathlib.Path, help="dir of led_*.npy + geometry.json")
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--out", default="fpm_validation.png")
    a = ap.parse_args()

    if a.captures:
        imgs, shifts, pupil, hr, truth = load_captures(a.captures)
        print(f"loaded {imgs.shape[0]} captures, {imgs.shape[1:]} px")
    else:
        imgs, shifts, pupil, hr, truth = make_synthetic()
        print(f"synthetic: {imgs.shape[0]} LED images (noisy), truth known")

    bf = int(jnp.argmin(jnp.sum(shifts ** 2, axis=1)))
    bf_img = jax.image.resize(jnp.sqrt(jnp.clip(imgs[bf], 0)), hr, "bilinear")
    rec = fpm.reconstruct_fpm(imgs, shifts, pupil, hr, steps=a.steps)

    print(f"LEDs used: {shifts.shape[0]}  |  max shift {int(jnp.abs(shifts).max())} px  "
          f"|  reconstruction {hr[0]}x{hr[1]}")
    if truth is not None:
        print(f"correlation  brightfield={_corr(bf_img, truth):.3f}   FPM={_corr(rec, truth):.3f}")

    fig, ax = plt.subplots(1, 3, figsize=(11, 4))
    for x, (im, t) in zip(ax, [(jnp.abs(bf_img), "brightfield (1 LED)"),
                               (jnp.abs(rec), "FPM (synthetic aperture)"),
                               (jnp.abs(truth) if truth is not None else jnp.abs(rec),
                                "ground truth" if truth is not None else "FPM")]):
        x.imshow(im, cmap="gray"); x.set_title(t); x.axis("off")
    fig.tight_layout(); fig.savefig(a.out, dpi=110)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
