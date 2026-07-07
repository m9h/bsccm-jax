"""Emit the optics-driven parameter block for led_head.scad.

Keeps the CAD's `panel_height` in sync with the DPC calibration: the illumination
height is computed from the objective NA and LED pitch by the same
``recommend_height_mm`` the reconstruction uses — one source of truth from optics
to printable part.

    python hardware/gen_led_head.py --na 0.25 --pitch 2.5 --n 32
"""

from __future__ import annotations

import argparse

from bsccm_jax import led_array


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--na", type=float, required=True, help="objective NA")
    ap.add_argument("--pitch", type=float, default=2.5, help="LED pitch (mm)")
    ap.add_argument("--n", type=int, default=32, help="LEDs per side")
    ap.add_argument("--rings", type=float, default=8.0, help="LED rings in NA cone")
    ap.add_argument("--px", type=float, default=0.5, help="sample-plane pixel (um)")
    ap.add_argument("--wavelength", type=float, default=0.525, help="LED wl (um)")
    a = ap.parse_args()

    h = led_array.recommend_height_mm(a.na, a.pitch, a.rings)
    arr = led_array.LEDArray(n=a.n, pitch_mm=a.pitch, height_mm=round(h, 1))
    rep = led_array.calibration_report(arr, a.na, a.wavelength, a.px, (256, 256))

    print("// --- paste into led_head.scad [Optics-driven] / [LED panel] ---")
    print(f"panel_height = {h:.1f};   // NA={a.na}, {a.n}x{a.n} @ {a.pitch}mm pitch")
    print(f"led_n     = {a.n};")
    print(f"led_pitch = {a.pitch};")
    print("//")
    print("// calibration check:")
    for k, v in rep.items():
        print(f"//   {k}: {v}")


if __name__ == "__main__":
    main()
