"""Real-cell DPC reconstruction on the BSCCM tiny subset.

Loads the four half-annulus intensity images per cell, reconstructs quantitative
phase with our JAX WOTF pipeline (Tikhonov + SCICO TV), and validates against the
dataset's own precomputed `DPC` channel. Renders a montage + reports correlation.
"""

import argparse

import jax.numpy as jnp
import numpy as np

from bsccm import BSCCM
from bsccm_jax import dpc


def corr(a, b):
    a = np.asarray(a, float).ravel(); b = np.asarray(b, float).ravel()
    a = a - a.mean(); b = b - b.mean()
    return float((a * b).sum() / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/BSCCM-tiny")
    ap.add_argument("--n", type=int, default=4, help="number of cells")
    ap.add_argument("--out", default="real_cell_dpc.png")
    args = ap.parse_args()

    data = BSCCM(args.data, cache_index=True)
    idxs = [int(i) for i in data.get_indices()[: args.n]]
    print(f"reconstructing {len(idxs)} cells: {idxs}")

    rows = []
    for idx in idxs:
        imgs = {k: np.asarray(data.read_image(idx, f"DPC_{k}", copy=True), float)
                for k in ("Top", "Bottom", "Left", "Right")}
        p_tik = dpc.reconstruct_dpc_2axis(imgs)
        p_tv = dpc.reconstruct_dpc_2axis(imgs, tv=True, max_iter=80)
        ref = np.asarray(data.read_image(idx, "DPC", copy=True), float)  # dataset's own DPC
        fl = np.asarray(data.read_image(idx, data.fluor_channel_names[3], copy=True), float)
        # orient to the reference (DPC sign is convention-dependent)
        s = np.sign(corr(p_tik, ref)) or 1.0
        p_tik, p_tv = s * np.asarray(p_tik), s * np.asarray(p_tv)
        c_tik, c_tv = corr(p_tik, ref), corr(p_tv, ref)
        print(f"  cell {idx:4d}: corr(ours_Tikhonov, precomputed DPC)={c_tik:+.3f} | TV={c_tv:+.3f}")
        rows.append((idx, imgs["Top"], p_tik, p_tv, ref, fl, c_tik, c_tv))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cols = ["raw DPC_Top", "ours: phase (Tikhonov)", "ours: phase (SCICO TV)",
            "dataset precomputed DPC", "fluorescence (matched)"]
    fig, ax = plt.subplots(len(rows), 5, figsize=(15, 3.1 * len(rows)))
    ax = np.atleast_2d(ax)
    for r, (idx, top, ptik, ptv, ref, fl, ct, cv) in enumerate(rows):
        panels = [(top, "gray"), (ptik, "viridis"), (ptv, "viridis"),
                  (ref, "viridis"), (fl, "magma")]
        for c, (img, cm) in enumerate(panels):
            ax[r, c].imshow(img, cmap=cm); ax[r, c].axis("off")
            if r == 0:
                ax[r, c].set_title(cols[c], fontsize=10)
        ax[r, 1].set_ylabel(f"cell {idx}")
        ax[r, 2].set_title(f"{cols[2]}\ncorr vs dataset = {cv:+.3f}", fontsize=9)
    fig.suptitle("BSCCM real-cell DPC: raw half-annulus intensity -> our JAX quantitative phase "
                 "(validated vs dataset's own DPC)", fontsize=12)
    fig.tight_layout()
    fig.savefig(args.out, dpi=110)
    mean_tv = np.mean([r[7] for r in rows])
    print(f"\nwrote {args.out} | mean corr(our TV phase, dataset DPC) = {mean_tv:+.3f}")


if __name__ == "__main__":
    main()
