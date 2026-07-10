"""Evaluate the Allen brightfield->lamin model on held-out files (PCC vs floor).

Independent signal-rich confirmation of the VSCyto2D conclusion: on a structural
target (lamin-B1 nuclear envelope), does our TRAINED pipeline beat the mean-image
floor by a wide margin (unlike BSCCM's surface markers, which sat at the floor)?

    /work/vsvenv/bin/python scripts/eval_allen.py --ckpt <best.ckpt> \
        --data /work/allen/allen_val.zarr --json /work/allen/metrics.json
"""

import argparse
import json

import numpy as np
import torch
from iohub import open_ome_zarr

from cytoland.engine import VSUNet
from viscy_utils.losses import MixedLoss

MODEL_CONFIG = dict(in_channels=1, out_channels=1, encoder_blocks=[3, 3, 9, 3],
                    dims=[96, 192, 384, 768], decoder_conv_blocks=2,
                    stem_kernel_size=[1, 2, 2], in_stack_depth=1, pretraining=False)


def pcc(a, b):
    a = np.asarray(a, np.float64).ravel(); b = np.asarray(b, np.float64).ravel()
    a = (a - a.mean()) / (a.std() + 1e-8); b = (b - b.mean()) / (b.std() + 1e-8)
    return float((a * b).mean())


def tiled(model, x, dev, tile=512, halo=64):
    H, W = x.shape
    out = np.zeros((H, W), np.float32)
    for y0 in range(0, H, tile):
        for x0 in range(0, W, tile):
            ya, yb = max(0, y0 - halo), min(H, y0 + tile + halo)
            xa, xb = max(0, x0 - halo), min(W, x0 + tile + halo)
            patch = torch.from_numpy(x[ya:yb, xa:xb])[None, None, None].to(dev)
            with torch.no_grad():
                p = model(patch).float().cpu().numpy()[0, 0, 0]
            yi0, xi0 = y0 - ya, x0 - xa
            hh, ww = min(tile, H - y0), min(tile, W - x0)
            out[y0:y0 + hh, x0:x0 + ww] = p[yi0:yi0 + hh, xi0:xi0 + ww]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", default="/work/allen/allen_val.zarr")
    ap.add_argument("--json", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--n", type=int, default=40)
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = VSUNet(architecture="fcmae", model_config=MODEL_CONFIG,
                   loss_function=MixedLoss(l1_alpha=1.0))
    sd = torch.load(args.ckpt, weights_only=True, map_location="cpu")["state_dict"]
    model.load_state_dict(sd)
    model.eval().to(dev)
    print(f"loaded Allen model on {dev}")

    plate = open_ome_zarr(args.data, mode="r")
    ch = plate.channel_names
    bi, li = ch.index("Brightfield"), ch.index("Lamin")
    preds, truths, rows = [], [], []
    for name, pos in list(plate.positions())[: args.n]:
        arr = np.asarray(pos["0"])[0]                       # (c, z, y, x)
        bright = arr[bi, 0].astype(np.float32)
        st = pos.zattrs["normalization"]["Brightfield"]["fov_statistics"]
        x = (bright - st["median"]) / (st["iqr"] + 1e-8)
        pred = tiled(model, x, dev)
        truth = arr[li, 0].astype(np.float32)
        preds.append(pcc(pred, truth)); truths.append(truth)
        rows.append((name, bright, pred, truth))

    mean_img = np.mean(truths, axis=0)
    floor = [pcc(mean_img, t) for t in truths]
    overall, fl = float(np.mean(preds)), float(np.mean(floor))
    print(f"\nAllen brightfield->lamin  (n={len(preds)} held-out FOVs):")
    print(f"  PCC {overall:+.3f}   mean-image floor {fl:+.3f}   (model beats floor by {overall-fl:+.3f})")
    print(f"  paper (3D fnet) lamin r ~0.85; this is a 2D single-slice variant")

    if args.json:
        json.dump({"pcc": round(overall, 4), "floor": round(fl, 4),
                   "beats_floor": bool(overall > fl), "margin": round(overall - fl, 4),
                   "n_fov": len(preds)}, open(args.json, "w"), indent=2)
        print("wrote", args.json)

    if args.out:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        n = min(3, len(rows))
        fig, ax = plt.subplots(n, 3, figsize=(11, 3.6 * n))
        ax = np.atleast_2d(ax)
        for r in range(n):
            name, bright, pred, truth = rows[r]
            for cc, (im, t, cm) in enumerate([(bright, "brightfield (in)", "gray"),
                                              (pred, "predicted lamin", "magma"),
                                              (truth, "true lamin", "magma")]):
                ax[r, cc].imshow(im, cmap=cm); ax[r, cc].axis("off")
                if r == 0:
                    ax[r, cc].set_title(t, fontsize=10)
        fig.suptitle("Allen Label-Free: brightfield -> lamin-B1 nuclear envelope", fontsize=12)
        fig.tight_layout(); fig.savefig(args.out, dpi=100)
        print("wrote", args.out)


if __name__ == "__main__":
    main()
