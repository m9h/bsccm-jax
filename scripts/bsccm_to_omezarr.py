"""Convert BSCCM cells into paired HCS OME-Zarr for VisCy/Cytoland fine-tuning.

Each cell becomes one FOV with channels:
  - Phase       : our DPC 2-axis reconstruction (label-free SOURCE)
  - Brightfield : raw brightfield (alternative label-free source)
  - Fluor_*     : the matched fluorescence bands (virtual-staining TARGETS)

VisCy's HCSDataModule reads this by channel name (source_channel / target_channel).
Cells are split across two wells (A/1 = train, A/2 = val) so the data module's
position-based splitting is trivial.

    uv run python scripts/bsccm_to_omezarr.py --data data/BSCCM-tiny \
        --out data/bsccm_vs.zarr --n 800 --val-frac 0.2
"""

import argparse

import numpy as np
from iohub import open_ome_zarr

from bsccm import BSCCM
from bsccm_jax import dpc


def _norm(a):
    a = np.asarray(a, np.float32)
    lo, hi = np.percentile(a, 1), np.percentile(a, 99)
    return (a - lo) / (hi - lo + 1e-6)


def convert(data_root, out_path, n_cells, val_frac):
    data = BSCCM(data_root, cache_index=True)
    idxs = [int(i) for i in data.get_indices()[:n_cells]]
    fluor = list(data.fluor_channel_names)
    channels = ["Phase", "Brightfield"] + fluor
    n_val = max(1, int(len(idxs) * val_frac))

    with open_ome_zarr(out_path, layout="hcs", mode="w", channel_names=channels) as plate:
        for j, idx in enumerate(idxs):
            imgs = {k: np.asarray(data.read_image(idx, f"DPC_{k}", copy=True), float)
                    for k in ("Top", "Bottom", "Left", "Right")}
            phase = np.asarray(dpc.reconstruct_dpc_2axis(imgs), np.float32)
            bf = _norm(data.read_image(idx, "Brightfield", copy=True))
            fl = [_norm(data.read_image(idx, c, copy=True)) for c in fluor]
            arr = np.stack([phase, bf, *fl]).astype(np.float32)   # (C, Y, X)
            arr = arr[None, :, None]                              # (T=1, C, Z=1, Y, X)

            well = "2" if j < n_val else "1"                      # A/2 = val, A/1 = train
            pos = plate.create_position("A", well, str(idx))
            pos.create_image("0", arr)

    print(f"wrote {out_path}: {len(idxs)} cells, {len(channels)} channels "
          f"({len(idxs) - n_val} train / {n_val} val)")
    print(f"channels: {channels}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/BSCCM-tiny")
    ap.add_argument("--out", default="data/bsccm_vs.zarr")
    ap.add_argument("--n", type=int, default=800)
    ap.add_argument("--val-frac", type=float, default=0.2)
    args = ap.parse_args()
    convert(args.data, args.out, args.n, args.val_frac)


if __name__ == "__main__":
    main()
