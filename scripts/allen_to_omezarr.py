"""Convert the Allen Label-Free lamin_b1 CZIs into VisCy OME-Zarr (brightfield->lamin).

Independent signal-rich virtual-staining benchmark (Ounkomol et al. 2018): predict
the lamin-B1 nuclear-envelope fluorescence from transmitted-light brightfield. Each
100X .czi is a (C=4, Z=70, Y, X) stack; channel order (confirmed visually) is
[CMDRP membrane, EGFP lamin, H3342 DNA, TL brightfield]. We take a few in-focus
z-slices per stack as 2D samples (brightfield[z] -> lamin[z]) and split BY FILE into
train/val zarrs so no cell leaks across the split.

    /work/vsvenv/bin/python scripts/allen_to_omezarr.py --src /work/allen/lamin_b1 \
        --out-train /work/allen/allen_train.zarr --out-val /work/allen/allen_val.zarr
"""

import argparse
import glob
import os

import numpy as np
from aicspylibczi import CziFile
from iohub import open_ome_zarr

C_BRIGHT, C_LAMIN = 3, 1
CHANNELS = ["Brightfield", "Lamin"]
Z_SLICES = [28, 39, 50]                    # 2D mode: in-focus central slices -> 2D samples
STACK_DEPTH = 28                           # 3D/2.5D mode: FIXED centered z-depth per tile.
                                           # Fixed (not a range) so every volume is identical
                                           # even though CZIs vary (Z=50 vs 70) — else the norm
                                           # metadata's np.stack across positions fails.
TILE = 256                                 # store FOVs already tiled to the patch size:
                                           # VisCy validation delivers the whole FOV and
                                           # expects it == yx_patch_size, so FOV==tile keeps
                                           # train and val consistent (and yields more data)


def _tiles(im):
    """Non-overlapping TILE x TILE crops covering the FOV (drops the ragged edge)."""
    h, w = im.shape
    for y0 in range(0, h - TILE + 1, TILE):
        for x0 in range(0, w - TILE + 1, TILE):
            yield y0, x0


def load_pair(path):
    arr = np.squeeze(CziFile(path).read_image()[0])   # (4, Z, Y, X)
    if arr.ndim != 4 or arr.shape[0] < 4:
        return None
    return arr


def write_zarr(files, out_path, stack=False):
    if os.path.exists(out_path):
        import shutil; shutil.rmtree(out_path)
    plate = open_ome_zarr(out_path, layout="hcs", mode="w", channel_names=CHANNELS)
    n = 0
    for fi, f in enumerate(files):
        arr = load_pair(f)
        if arr is None:
            continue
        Z = arr.shape[1]
        if stack:                                          # 3D/2.5D: one z-volume per tile
            if Z < STACK_DEPTH:
                continue
            z0 = Z // 2 - STACK_DEPTH // 2                  # fixed centered depth (uniform)
            z1 = z0 + STACK_DEPTH
            for ti, (y0, x0) in enumerate(_tiles(arr[C_BRIGHT, z0])):
                vol = arr[[C_BRIGHT, C_LAMIN], z0:z1, y0:y0 + TILE, x0:x0 + TILE].astype(np.float32)
                pos = plate.create_position(str(fi), "0", str(ti))
                pos.create_image("0", vol[None])           # (T=1, C=2, Z, Y, X)
                n += 1
        else:                                              # 2D: single central slices
            for zi, z in enumerate(Z_SLICES):
                if z >= Z:
                    continue
                bright, lamin = arr[C_BRIGHT, z].astype(np.float32), arr[C_LAMIN, z].astype(np.float32)
                for ti, (y0, x0) in enumerate(_tiles(bright)):
                    bt = bright[y0:y0 + TILE, x0:x0 + TILE]
                    lt = lamin[y0:y0 + TILE, x0:x0 + TILE]
                    pos = plate.create_position(str(fi), str(zi), str(ti))
                    pos.create_image("0", np.stack([bt, lt])[None, :, None])   # (T,C,Z=1,Y,X)
                    n += 1
    plate.close()
    print(f"wrote {out_path}: {n} samples from {len(files)} files ({'3D' if stack else '2D'})")
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="/work/allen/lamin_b1")
    ap.add_argument("--out-train", default="/work/allen/allen_train.zarr")
    ap.add_argument("--out-val", default="/work/allen/allen_val.zarr")
    ap.add_argument("--n-val", type=int, default=20)
    ap.add_argument("--stack", action="store_true", help="store z-volumes for 2.5D/3D training")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.src, "*100X*.czi")))   # 100X only
    rng = np.random.default_rng(0)
    files = list(rng.permutation(files))
    val, train = files[: args.n_val], files[args.n_val:]
    print(f"{len(files)} 100X files -> {len(train)} train / {len(val)} val")
    write_zarr(train, args.out_train, stack=args.stack)
    write_zarr(val, args.out_val, stack=args.stack)


if __name__ == "__main__":
    main()
