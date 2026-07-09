"""Real-data FPM on the UConn Smart Imaging Lab datasets — an external oracle.

Validates our `fpm.py` EPRY reconstruction on genuinely-external experimental
LED-array captures (Zheng lab, Nat. Rev. Phys. 2021), with published geometry
and a reference MATLAB pipeline. Unlike BSCCM-coherent, these ship the full
physical model — LED pitch/height AND a glass-substrate refraction correction
(an iterative Snell's-law solve for the beam offset through the coverslip).

That glass term is the headline experiment here: our BSCCM-coherent recon had a
residual periodic *lattice* artifact that survived global geometric calibration.
Unmodeled substrate refraction bends every off-axis illumination angle slightly,
which is exactly the kind of sub-pixel, radius-dependent shift error that shows
up as a lattice. So we reconstruct BOTH ways — naive (angle = atan of LED
position) and glass-corrected — and put them side by side.

    PYTHONPATH=src python scripts/fpm_uconn.py --mat data/fpm_uconn/USAF_red.mat \
        --out fpm_uconn_usaf.png

Faithful ports of the repo's Funcs/LED_location.m and Funcs/k_vector.m.
"""

import argparse

import numpy as np
import scipy.io as sio

from bsccm_jax import fpm

# UConn rig constants (FP_recover_code.m)
XSTART, YSTART, ARRAYSIZE = 18, 20, 15
H_MM, LEDP_MM, NGLASS, T_MM = 90.88, 4.0, 1.52, 1.0
NA_OBJ, SPSIZE_M, UPSMP = 0.1, 1.845e-6, 4


def led_location(xstart=XSTART, ystart=YSTART, arraysize=ARRAYSIZE):
    """Spiral-out LED lighting sequence (port of LED_location.m)."""
    node = np.zeros(70, int); node[0] = 1; dif = 1; judge = 1
    for i in range(1, 70):
        node[i] = node[i - 1] + dif
        if judge < 2:
            judge += 1
        else:
            dif += 1; judge = 1
    n = arraysize ** 2
    xl = np.zeros(n, int); yl = np.zeros(n, int)
    xl[0], yl[0] = xstart, ystart
    xy_order = 2
    for i in range(1, n):
        if i + 1 > node[xy_order - 1]:      # 1-based node index -> 0-based
            xy_order += 1
        if xy_order % 2 == 0:
            xl[i] = xl[i - 1] + (-1) ** ((xy_order // 2) % 2 + 1)
            yl[i] = yl[i - 1]
        else:
            xl[i] = xl[i - 1]
            yl[i] = yl[i - 1] + (-1) ** (((xy_order - 1) // 2) % 2 + 1)
    return xl, yl


def _angle_naive(x0, y0, H):
    l = np.hypot(x0, y0)
    theta = np.arctan2(l, H)              # no substrate
    return np.abs(np.sin(theta))


def _angle_glass(x0, y0, H, h, n):
    """Illumination NA through a glass slide (port of k_vector.m::calculate)."""
    l = np.hypot(x0, y0)
    xoff = 0.0; xint = 1.0
    while abs(xint) > 1e-3:
        thetag = -np.arcsin((l - xoff) / np.sqrt((l - xoff) ** 2 + H ** 2) / n)
        xint = xoff + h * np.tan(thetag)
        xoff = xoff - xint
    theta = np.arcsin((l - xoff) / np.sqrt((l - xoff) ** 2 + H ** 2))
    return np.abs(np.sin(theta))


def k_vector(xi, yi, theta_deg, xint, yint, glass=True):
    """(kx,ky) normalized illumination-NA components per LED (port of k_vector.m)."""
    kx = np.zeros(len(xi)); ky = np.zeros(len(xi)); nat = np.zeros(len(xi))
    for t in range(len(xi)):
        x0 = xint + xi[t] * LEDP_MM
        y0 = yint + yi[t] * LEDP_MM
        x1 = x0 * np.cos(np.deg2rad(theta_deg)) - y0 * np.sin(np.deg2rad(theta_deg))
        y1 = x0 * np.sin(np.deg2rad(theta_deg)) + y0 * np.cos(np.deg2rad(theta_deg))
        thetal = np.arctan2(y1, x1)
        na = _angle_glass(x1, y1, H_MM, T_MM, NGLASS) if glass else _angle_naive(x1, y1, H_MM)
        nat[t] = na
        kx[t] = -na * np.cos(thetal)
        ky[t] = -na * np.sin(thetal)
    return kx, ky, nat


def reconstruct(imgs, kx, ky, wl, df, pupil_r, hr):
    shifts = np.stack([np.round(ky / wl / df), np.round(kx / wl / df)], 1).astype(int)
    pupil = fpm.circ_pupil(imgs.shape[1:], pupil_r)
    rec, _P, _s = fpm.reconstruct_fpm_epry(imgs, shifts, pupil, (hr, hr),
                                           iters=10, update_pupil=True, correct_positions=False)
    return np.asarray(rec), shifts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mat", default="data/fpm_uconn/USAF_red.mat")
    ap.add_argument("--out", default="fpm_uconn.png")
    args = ap.parse_args()

    m = sio.loadmat(args.mat, squeeze_me=True)
    stack = np.asarray(m["imlow_HDR"], np.float32)          # (H, W, nLED)
    imgs = np.transpose(stack, (2, 0, 1))                    # (nLED, H, W)
    wl = float(m["wlength"]); theta = float(m["theta"])
    xint = float(m["xint"]); yint = float(m["yint"])
    N = imgs.shape[1]; hr = N * UPSMP
    df = 1.0 / (N * SPSIZE_M)                                # LR-spectrum sampling [cyc/m/px]
    pupil_r = NA_OBJ / wl / df

    xl, yl = led_location()
    xi, yi = xl - XSTART, yl - YSTART
    kx_g, ky_g, nat_g = k_vector(xi, yi, theta, xint, yint, glass=True)
    kx_n, ky_n, _ = k_vector(xi, yi, theta, xint, yint, glass=False)
    print(f"{imgs.shape[0]} LEDs, LR {N}px, wl {wl*1e9:.0f}nm, NA_obj {NA_OBJ}, "
          f"pupil_r {pupil_r:.1f}px, max illum NA {nat_g.max():.3f} (synthetic NA "
          f"{NA_OBJ + nat_g.max():.2f} -> ~{(NA_OBJ+nat_g.max())/NA_OBJ:.1f}x)")

    rec_g, sh_g = reconstruct(imgs, kx_g, ky_g, wl, df, pupil_r, hr)
    rec_n, sh_n = reconstruct(imgs, kx_n, ky_n, wl, df, pupil_r, hr)
    print(f"glass vs naive max |shift| diff: {np.abs(sh_g - sh_n).max()} px "
          f"(mean {np.abs(sh_g - sh_n).mean():.2f})")

    import jax, jax.numpy as jnp
    bf = np.asarray(jax.image.resize(jnp.asarray(np.sqrt(imgs[0])), (hr, hr), "bilinear"))
    S = lambda x: np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(x))))
    yy, xx = np.mgrid[0:hr, 0:hr] - hr // 2
    r = np.hypot(yy, xx); cutoff = pupil_r * UPSMP
    beyond = lambda im: float(S(im)[r > cutoff].sum() / S(im).sum())
    print(f"spectral energy beyond objective cutoff:  brightfield {beyond(bf):.3f}  "
          f"FPM-naive {beyond(np.abs(rec_n)):.3f}  FPM-glass {beyond(np.abs(rec_g)):.3f}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cr = slice(50, hr - 50)
    fig, ax = plt.subplots(2, 3, figsize=(13, 9))
    panels = [
        (bf[cr, cr], "brightfield (1 LED)", "gray"),
        (np.abs(rec_n)[cr, cr], "FPM amplitude — NAIVE angle", "gray"),
        (np.abs(rec_g)[cr, cr], "FPM amplitude — GLASS-corrected", "gray"),
        (S(bf), "brightfield spectrum", "magma"),
        (S(np.abs(rec_n)), "FPM spectrum — naive", "magma"),
        (S(np.abs(rec_g)), "FPM spectrum — glass", "magma"),
    ]
    for a, (im, t, cm) in zip(ax.ravel(), panels):
        a.imshow(im, cmap=cm); a.axis("off"); a.set_title(t, fontsize=10)
    fig.suptitle(f"UConn real-data FPM ({imgs.shape[0]} LEDs, {wl*1e9:.0f}nm, NA {NA_OBJ}): "
                 f"glass-substrate refraction correction vs naive geometry", fontsize=12)
    fig.tight_layout(); fig.savefig(args.out, dpi=110)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
