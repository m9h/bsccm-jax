"""Real-data FPM on BSCCM-coherent — validates fpm.py on actual LED-array data.

Each coherent cell is imaged under ~566 single-LED illuminations at known
quasi-dome positions (x,y,z). We map each LED to its pupil shift in the high-res
spectrum, run the differentiable FPM reconstruction (fpm.py), and compare against
the on-axis (brightfield) image to show synthetic-aperture super-resolution — on
real data, not a phantom.

    PYTHONPATH=src python scripts/fpm_bsccm_coherent.py --n-led 400 --out fpm_coherent.png
"""

import argparse
import glob
import json
import os

import numpy as np

from bsccm_jax import fpm


def led_na_map():
    """led_num -> (na_y, na_x) using BSCCM's OWN calibration (z_offset=8).

    Using the raw nominal z=50 gives mis-scaled shifts -> a periodic lattice
    artifact in the reconstruction; the calibrated NA fixes it.
    """
    import bsccm
    from bsccm.led_array_calibration import load_led_positions_from_json
    qd_path = os.path.join(os.path.dirname(bsccm.__file__), "quasi_dome_design.json")
    naxy, _na, _cart = load_led_positions_from_json(qd_path)      # z_offset=8 default
    qd = json.load(open(qd_path))
    # naxy[:,0] from x (na_x), naxy[:,1] from y (na_y)
    return {e["led_num"]: (float(naxy[i, 1]), float(naxy[i, 0])) for i, e in enumerate(qd["led_list"])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", type=int, default=None)
    ap.add_argument("--n-led", type=int, default=400, help="use the N most on-axis LEDs")
    ap.add_argument("--hr", type=int, default=256)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--out", default="fpm_coherent.png")
    args = ap.parse_args()

    from bsccm import BSCCM
    root = glob.glob("/mnt/t9/bsccm_coherent/extracted/*/")[0]
    data = BSCCM(root, cache_index=True)
    la = data.global_metadata["led_array"]
    dx = float(la["pixel_size_um"]); wl = la["wavelength_nm"] / 1000.0
    na = la["objective"]["NA"]
    chans = data.led_array_channel_names
    idx = int(args.cell) if args.cell is not None else int(data.get_indices()[0])

    namap = led_na_map()
    lr = None; imgs = []; shifts = []
    df = None
    for c in chans:
        num = int(c.split("_")[1])
        if num not in namap:
            continue
        na_y, na_x = namap[num]                               # calibrated illumination NA
        im = np.asarray(data.read_image(idx, c, copy=True), np.float32)
        if lr is None:
            lr = im.shape; df = 1.0 / (lr[0] * dx)            # HR-spectrum sampling [cyc/um/px]
        sy, sx = round((na_y / wl) / df), round((na_x / wl) / df)
        imgs.append(im); shifts.append((sy, sx, na_y ** 2 + na_x ** 2))
    # keep the N most on-axis LEDs (well-conditioned coverage)
    order = np.argsort([s[2] for s in shifts])[: args.n_led]
    imgs = np.stack([imgs[i] for i in order])
    shifts = np.asarray([[shifts[i][0], shifts[i][1]] for i in order], int)
    pupil_r = (na / wl) / df
    pupil = fpm.circ_pupil(lr, pupil_r)
    print(f"cell {idx}: {len(imgs)} LEDs, LR {lr}, dx {dx}um, NA {na}, pupil_r {pupil_r:.1f}px, "
          f"max|shift| {np.abs(shifts).max()}px, HR {args.hr}")

    rec, _pupil, _sh = fpm.reconstruct_fpm_epry(imgs, shifts, pupil, (args.hr, args.hr),
                                                iters=max(10, args.steps // 30), correct_positions=True)

    # brightfield = the most on-axis LED, upsampled; FPM should be sharper
    import jax, jax.numpy as jnp
    bf = np.asarray(jax.image.resize(jnp.asarray(np.sqrt(imgs[0])), (args.hr, args.hr), "bilinear"))
    # spectral support: FPM spectrum should extend beyond the objective cutoff
    S_bf = np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(bf))))
    S_fpm = np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(np.asarray(rec)))))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 4, figsize=(15, 3.8))
    for a, (img, t, cm) in zip(ax, [
            (bf, "brightfield (1 LED)", "gray"),
            (np.abs(rec), "FPM amplitude", "gray"),
            (np.angle(rec), "FPM phase", "twilight"),
            (S_fpm, "FPM spectrum (extended aperture)", "magma")]):
        a.imshow(img, cmap=cm); a.axis("off"); a.set_title(t, fontsize=10)
    fig.suptitle(f"BSCCM-coherent real-data FPM: {len(imgs)} single-LED images -> super-resolved "
                 f"complex object (NA {na}, {wl*1000:.0f}nm)", fontsize=11)
    fig.tight_layout(); fig.savefig(args.out, dpi=110)
    # simple resolution proxy: fraction of spectral energy beyond the objective cutoff
    h, w = args.hr, args.hr; yy, xx = np.mgrid[0:h, 0:w] - np.array([h // 2, w // 2])[:, None, None]
    r = np.sqrt(yy ** 2 + xx ** 2); cutoff = pupil_r * (h / lr[0])
    beyond = lambda S: float(S[r > cutoff].sum() / S.sum())
    print(f"spectral energy beyond objective cutoff:  brightfield {beyond(S_bf):.3f}  FPM {beyond(S_fpm):.3f}")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
