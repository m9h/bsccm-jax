# LED-matrix illumination head — DPC/FPM on OpenFlexure

A 3D-printable head that holds a programmable LED matrix above the sample for
**differential phase contrast (DPC)** and, later, **Fourier ptychography (FPM)**.
The illumination geometry is driven by the same calibration the reconstruction
uses, so the optics, the CAD, and the forward model stay consistent:

```
bsccm_jax.led_array.recommend_height_mm(NA, pitch)  ->  panel_height  ->  led_head*.scad
bsccm_jax.led_array.build_dpc_forward_from_array(...)  <-  as-built height + NA + pixel size
```

You do **not** need a phase-sensitive camera — DPC recovers phase computationally
from ordinary intensity images. Capture linear **raw** with fixed exposure/gain/WB.

## Files

| File | What |
|---|---|
| `led_frame.scad` | shared panel frame + parameters (no geometry on its own) |
| `led_head.scad` | generic head on a 3-screw base ring (no OpenFlexure dependency) |
| `led_head_ofm.scad` | head that **clips onto the stock OpenFlexure condenser dovetail** |
| `gen_led_head.py` | compute `panel_height` + the LED calibration from the objective NA |
| `ofmlibs/` | vendored OpenFlexure dovetail libs (CERN-OHL-W v2 — see NOTICE) |

## Workflow

1. **Compute the geometry for your objective:**
   ```
   uv run python hardware/gen_led_head.py --na 0.25 --pitch 2.5 --n 32
   ```
   Paste the emitted `panel_height` / `led_n` / `led_pitch` into `led_frame.scad`.
   (NA 0.25, 32×32 @ 2.5 mm → `panel_height = 77.5`, 208 bright-field LEDs,
   corner NA 0.577 → FPM headroom to ~0.83.)

2. **Render the STL** (pick the head you want):
   ```
   openscad -o led_head_ofm.stl hardware/led_head_ofm.scad   # clips onto OpenFlexure
   openscad -o led_head.stl     hardware/led_head.scad       # generic post stand
   ```

3. **Slice (OrcaSlicer):** black PETG (kills stray internal reflections; PLA fine
   for a first test), 0.2 mm layers, 3–4 walls, 15–20 % infill, tree supports
   (only the internal PCB ledge overhangs), 5 mm brim. Import as-is — OrcaSlicer
   drops it onto the plate.

4. **Assemble:** drop the panel in **LED-face-down** (it rests on the border
   ledge, aperture below); tune `pcb_clear` and reprint if tight. Add a *weak*
   diffuser in the side slot only if you see per-LED hotspots — keep angular
   selectivity, so err toward none.

5. **Mount + measure:** the OpenFlexure illumination dovetail is a *sliding* fit,
   so slide the head to the height you want, then **measure the true panel-to-
   sample distance** and pass it as `LEDArray(height_mm=<measured>)`. The
   reconstruction adapts to the as-built geometry — the CAD height is only a target.

6. **Capture + reconstruct:**
   ```python
   from bsccm_jax import led_array
   arr = led_array.LEDArray(n=32, pitch_mm=2.5, height_mm=<measured>)
   masks = led_array.dpc_led_masks(arr, na=0.25)          # which LEDs per frame -> illuminate
   fwd   = led_array.build_dpc_forward_from_array(arr, shape, na=0.25, pixel_size_um=<measured>)
   # capture 4 raw frames under masks, then dpc.reconstruct_scico(images, fwd)
   ```

## Bill of materials

- APA102/DotStar **32×32** RGB LED matrix (~$50–70) — APA102 so `illuminate` drives it.
- **Teensy 4.0** + Zack Phillips' `illuminate` firmware (~$25) — has native DPC/annulus commands.
- Camera: your **imx477 HQ in raw mode** (or imx219/imx708), one colour channel matched to green LEDs.
- Optional weak diffuser sheet; M3 hardware for the generic base ring.

## Licensing

`led_frame.scad`, `led_head.scad`, `gen_led_head.py`, and the Python package are
under the repository's license (BSD-3). **`led_head_ofm.scad` and everything in
`ofmlibs/` incorporate OpenFlexure geometry and are covered by CERN-OHL-W v2**
((c) Richard Bowman / OpenFlexure) — see `ofmlibs/NOTICE`. Keep that file's
modifications shareable under CERN-OHL-W if you distribute the OFM head.
