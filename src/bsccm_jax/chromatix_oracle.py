"""Chromatix wave-optics oracle for the DPC forward model (Tier 3).

Our `dpc.py` forward is the *linearized* weak-object transfer function (WOTF).
Chromatix (Deb, Both, ... Waller ... Turaga, Nature Methods 2026;
github.com/TuragaLab/chromatix) is a full differentiable wave-optics simulator.
For a weak phase object the two must agree — so Chromatix is a JAX-vs-JAX oracle
that validates our hand-rolled WOTF without leaving the stack.

Model: partially-coherent DPC. Each source point in a half-annulus is a tilted
plane wave (Chromatix `plane_wave(..., kykx=...)`); it passes through the thin
phase sample (`phase_change`); the objective imposes its NA as a coherent
low-pass in the pupil plane; intensities are summed incoherently over the source.

STATUS: WIP. Chromatix is integrated (installs on jax 0.10.2, coexists with our
stack) and this forward runs. Calibration so far:
  * FIXED: kykx is the angular wavenumber 2*pi*sin(theta)/wavelength (was missing
    the 2*pi -> illumination tilt 2pi x too small -> no DPC asymmetry).
  * STILL OPEN: cross-check with the WOTF is still ~0 correlation, so a second
    convention mismatch remains. Prime suspect: mixing Chromatix's field/FFT grid
    with our shifted F/iF pupil (frequency grids may not align), or the objective
    should be modelled with Chromatix's own ff_lens + circular_pupil (4f chain)
    rather than our hand-applied Fourier low-pass. Diagnose by comparing the raw
    per-angle |field|^2 spectra between the two before trusting this as an oracle.
"""

from __future__ import annotations

import chromatix.functional as cf
import jax
import jax.numpy as jnp
import numpy as np

from bsccm_jax import dpc

F, iF = dpc.F, dpc.iF


def _half_annulus_points(na_illum, wavelength, half, n, na_inner=0.0):
    """Source-point angular wavevectors (ky, kx) filling a half of the illum NA.

    Chromatix builds the plane wave as exp(1j * kykx . grid) with grid in um, so
    kykx is the ANGULAR transverse wavenumber 2*pi*sin(theta)/wavelength (not the
    plain spatial frequency sin(theta)/wavelength — the 2*pi matters).
    """
    sin_th = np.sqrt(np.random.default_rng(0).uniform(na_inner ** 2, na_illum ** 2, n * 3))
    r = 2.0 * np.pi * sin_th / wavelength
    th = np.random.default_rng(1).uniform(0, 2 * np.pi, n * 3)
    ky, kx = r * np.sin(th), r * np.cos(th)
    keep = {"top": ky > 0, "bottom": ky < 0, "left": kx < 0, "right": kx > 0}[half]
    ky, kx = ky[keep][:n], kx[keep][:n]
    return np.stack([ky, kx], axis=1)


def dpc_forward_chromatix(phase, *, wavelength=0.532, dx=0.2, na=0.5,
                          na_illum=0.5, half="top", n_src=48):
    """Wave-optics DPC intensity under half-annulus illumination (Chromatix)."""
    shape = phase.shape
    pts = _half_annulus_points(na_illum, wavelength, half, n_src)  # concrete np
    pupil = (dpc.generate_pupil(wavelength, dx, na, shape)).real   # NA low-pass mask

    acc = jnp.zeros(shape)
    for ky, kx in pts:                             # ~n_src concrete source points
        field = cf.plane_wave(shape, dx, wavelength, kykx=(float(ky), float(kx)))
        field = cf.phase_change(field, phase)
        u = field.u.squeeze()                      # complex field at sample
        u = iF(F(u) * pupil)                        # objective NA (pupil low-pass)
        acc = acc + (jnp.abs(u) ** 2).real
    return acc / len(pts)


def dpc_measurement_chromatix(phase, **kw):
    """Normalized 2-axis differential DPC via the Chromatix forward."""
    def diff(a, b):
        s = a + b
        return jnp.where(s > 0, (a - b) / s, 0.0)
    T = dpc_forward_chromatix(phase, half="top", **kw)
    B = dpc_forward_chromatix(phase, half="bottom", **kw)
    L = dpc_forward_chromatix(phase, half="left", **kw)
    R = dpc_forward_chromatix(phase, half="right", **kw)
    return jnp.stack([diff(T, B), diff(L, R)])
