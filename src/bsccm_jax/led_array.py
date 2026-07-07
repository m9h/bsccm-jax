"""Physical LED-matrix geometry for DPC on an OpenFlexure (or any) microscope.

Bridges a real programmable LED matrix — an APA102/DotStar panel mounted above
the sample and driven by Zack Phillips' ``illuminate`` firmware — to the WOTF DPC
forward model in :mod:`bsccm_jax.dpc`.

Physics.  An LED at lateral offset ``(x, y)`` (mm) and height ``h`` (mm) above the
sample illuminates it along direction cosines ``(a, b) = (x, y) / r`` with
``r = sqrt(x**2 + y**2 + h**2)``.  In the partially-coherent WOTF model that LED is
a *point source* in the pupil at spatial frequency ``(a, b) / wavelength``.  A DPC
frame lights every bright-field LED (``|(a, b)| <= NA``) lying on one side of a
diameter; the four defaults (0/180/90/270 deg) are the L/R/T/B half-cones that
:func:`dpc.annular_sources` models as continuous half-disks.

So one geometry yields two consistent things:

  * :func:`dpc_led_masks`   -> which physical LEDs to switch on per frame (hardware).
  * :func:`led_dpc_sources` -> the discrete-source stack for :func:`dpc.generate_wotf`
                               (a faithful forward model; ``annular_sources`` is its
                               dense-array limit).

Axis convention matches ``dpc.spatial_freq_grid``: the illumination x-axis maps to
``fx`` (array axis 0), y to ``fy`` (axis 1), and the DPC half-plane test is
``cos(rot) * a >= sin(rot) * b`` — identical to :func:`dpc.annular_sources`.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float

from .dpc import DPCForward, generate_pupil, generate_wotf, spatial_freq_grid

jax.config.update("jax_enable_x64", True)


@dataclass(frozen=True)
class LEDArray:
    """A flat square matrix of addressable LEDs, centred on the optical axis.

    n:         LEDs per side (e.g. 32 for a 32x32 panel).
    pitch_mm:  centre-to-centre LED spacing.
    height_mm: array-to-sample distance along the optical axis.
    """

    n: int
    pitch_mm: float
    height_mm: float

    def positions_mm(self) -> tuple[Float[Array, "n n"], Float[Array, "n n"]]:
        """(x, y) offsets from the optical axis for every LED."""
        c = (self.n - 1) / 2.0
        idx = (jnp.arange(self.n) - c) * self.pitch_mm
        return jnp.meshgrid(idx, idx, indexing="ij")

    def direction_cosines(self) -> tuple[Float[Array, "n n"], Float[Array, "n n"]]:
        """Illumination direction cosines (a, b) per LED (a=sin of x-tilt, etc.)."""
        x, y = self.positions_mm()
        r = jnp.sqrt(x**2 + y**2 + self.height_mm**2)
        return x / r, y / r

    def illumination_na(self) -> Float[Array, "n n"]:
        """Per-LED illumination NA, ``sqrt(a**2 + b**2)``."""
        a, b = self.direction_cosines()
        return jnp.sqrt(a**2 + b**2)

    def corner_na(self) -> float:
        """Max illumination NA (array corner) — the dark-field / FPM ceiling."""
        return float(self.illumination_na().max())


def recommend_height_mm(na: float, pitch_mm: float, rings: float = 8.0) -> float:
    """Height that fits ``rings`` LED pitches inside the objective-NA cone.

    The bright-field LEDs occupy a disk of radius ``R_NA = h * NA / sqrt(1 - NA**2)``
    on the array.  Requiring ``R_NA >= rings * pitch`` and solving for ``h`` gives a
    smooth DPC half-disk source; more rings (or finer pitch) -> smoother.
    """
    return float(rings * pitch_mm * jnp.sqrt(1.0 - na**2) / na)


def bright_cone_mask(array: LEDArray, na: float) -> Bool[Array, "n n"]:
    """LEDs inside the objective NA (bright field) — the DPC source support."""
    return array.illumination_na() <= na


def dpc_led_masks(
    array: LEDArray, na: float, rotations_deg=(0.0, 180.0, 90.0, 270.0)
) -> Bool[Array, "k n n"]:
    """Which LEDs to light per DPC frame: in-cone AND on one side of the diameter.

    Half-plane convention matches :func:`dpc.annular_sources`.
    """
    a, b = array.direction_cosines()
    cone = bright_cone_mask(array, na)
    masks = []
    for rot in rotations_deg:
        t = jnp.deg2rad(rot)
        half = (jnp.cos(t) * a) >= (jnp.sin(t) * b)
        masks.append(cone & half)
    return jnp.stack(masks)


def masks_to_led_indices(masks: Bool[Array, "k n n"]) -> list[list[tuple[int, int]]]:
    """Per-frame lists of ``(row, col)`` LED indices — feed to ``illuminate``/hardware."""
    frames = []
    for m in masks:
        rows, cols = jnp.nonzero(m)
        frames.append([(int(r), int(c)) for r, c in zip(rows, cols)])
    return frames


def led_dpc_sources(
    array: LEDArray,
    na: float,
    wavelength_um: float,
    pixel_size_um: float,
    shape,
    rotations_deg=(0.0, 180.0, 90.0, 270.0),
) -> Float[Array, "k h w"]:
    """Discrete-LED source stack on the pupil grid, for :func:`dpc.generate_wotf`.

    Each lit LED becomes a unit source at pupil coordinate ``(a, b) / wavelength``,
    snapped to the nearest spatial-frequency sample.  This is the faithful
    counterpart of the continuous :func:`dpc.annular_sources` (which is the
    dense-array limit).  Use it when the array is coarse enough that the discrete
    source structure matters.
    """
    fx_g, fy_g = spatial_freq_grid(shape, pixel_size_um)
    fx_ax, fy_ax = fx_g[:, 0], fy_g[0, :]  # 1-D freq axes (fx->axis0, fy->axis1)
    a, b = array.direction_cosines()
    masks = dpc_led_masks(array, na, rotations_deg)
    ix = jnp.argmin(jnp.abs(fx_ax[None, None, :] - (a / wavelength_um)[..., None]), -1)
    iy = jnp.argmin(jnp.abs(fy_ax[None, None, :] - (b / wavelength_um)[..., None]), -1)
    src = []
    for m in masks:
        img = jnp.zeros(shape, jnp.float64)
        img = img.at[ix[m], iy[m]].add(1.0)
        src.append(img)
    return jnp.stack(src)


def build_dpc_forward_from_array(
    array: LEDArray,
    shape,
    *,
    na: float,
    pixel_size_um: float,
    wavelength_um: float = 0.525,
    rotations_deg=(0.0, 180.0, 90.0, 270.0),
) -> DPCForward:
    """A :class:`dpc.DPCForward` whose sources are this array's *actual* lit LEDs.

    Drop-in for ``DPCForward.build`` when you want the forward model to reflect the
    discrete illumination you can physically produce, not the idealised half-disk.
    """
    pupil = generate_pupil(wavelength_um, pixel_size_um, na, shape)
    sources = led_dpc_sources(array, na, wavelength_um, pixel_size_um, shape, rotations_deg)
    Hu, Hp = generate_wotf(sources, pupil)
    return DPCForward(Hu=Hu, Hp=Hp)


def calibration_report(
    array: LEDArray, na: float, wavelength_um: float, pixel_size_um: float, shape
) -> dict:
    """Numbers you need to sanity-check the mount before capturing.

    Returns a dict (also nicely ``print``-able) covering how many LEDs fall in the
    bright-field cone, the angular sampling, whether the pupil is well sampled by
    the reconstruction grid, and the dark-field / FPM headroom.
    """
    cone = bright_cone_mask(array, na)
    n_bf = int(cone.sum())
    r_na_mm = float(array.height_mm * na / jnp.sqrt(1.0 - na**2))
    rings = r_na_mm / array.pitch_mm
    dalpha = array.pitch_mm / array.height_mm  # small-angle LED spacing
    # pupil radius in reconstruction pixels: fmax / df, df = 1/(N*px)
    df = 1.0 / (shape[0] * pixel_size_um)
    pupil_px = (na / wavelength_um) / df
    rep = {
        "objective_na": na,
        "bright_field_leds": n_bf,
        "na_cone_radius_mm": round(r_na_mm, 2),
        "led_rings_in_cone": round(float(rings), 2),
        "angular_sampling_deg": round(float(jnp.rad2deg(dalpha)), 3),
        "pupil_radius_px": round(float(pupil_px), 1),
        "corner_na_fpm_ceiling": round(array.corner_na(), 3),
    }
    if rings < 4:
        rep["warning"] = (
            f"only ~{rings:.1f} LED rings in the NA cone; raise height or use a "
            f"finer-pitch panel for a smooth DPC source"
        )
    return rep
