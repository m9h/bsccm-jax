"""VSCyto2D oracle — run the PUBLISHED Cytoland checkpoint on its own test data.

This is the diagnostic that disambiguates our BSCCM virtual-staining result
(which only matched a mean-image floor): is our VisCy pipeline sound, or is
BSCCM just an unusually mean-dominated dataset? Here we load the official
VSCyto2D checkpoint (compmicro-czb/VSCyto2D) and run it on the official A549
test set, reporting Pearson correlation (PCC) of predicted vs experimental
fluorescence per channel. If we reproduce the paper's PCC band (~0.6-0.85,
nuclei ~0.71 on their HEK comparison), our pipeline is validated and the BSCCM
outcome is confirmed as a property of BSCCM.

Model recipe (fcmae_2d.yml): FcmaeUNet, in=1 (Phase3D), out=2 (Nucl, Mem).
Input normalized by per-FOV median/iqr (matches vscyto2d/predict.yml).

    /work/vsvenv/bin/python scripts/eval_vscyto2d.py \
        --ckpt /work/vscyto2d/VSCyto2D.ckpt \
        --data /work/vscyto2d/a549_hoechst_cellmask_test.zarr --json /work/vscyto2d/metrics.json
"""

import argparse
import json

import numpy as np
import torch
from iohub import open_ome_zarr

from cytoland.engine import FcmaeUNet

MODEL_CONFIG = dict(in_channels=1, out_channels=2, encoder_blocks=[3, 3, 9, 3],
                    dims=[96, 192, 384, 768], decoder_conv_blocks=2,
                    stem_kernel_size=[1, 2, 2], in_stack_depth=1, pretraining=False)
TARGETS = ["Nucl", "Mem"]


def pcc(a, b):
    a = np.asarray(a, np.float64).ravel(); b = np.asarray(b, np.float64).ravel()
    a = (a - a.mean()) / (a.std() + 1e-8); b = (b - b.mean()) / (b.std() + 1e-8)
    return float((a * b).mean())


def build_model(ckpt):
    sd = torch.load(ckpt, weights_only=False, map_location="cpu")["state_dict"]
    # ckpt hparams are empty -> construct explicitly, then load weights.
    last_err = None
    for kw in ({"model_config": MODEL_CONFIG},
               {"model_config": MODEL_CONFIG, "fit_mask_ratio": 0.0}):
        try:
            m = FcmaeUNet(**kw)
            m.load_state_dict(sd)
            return m
        except Exception as e:  # noqa: BLE001
            last_err = e
    raise RuntimeError(f"could not build FcmaeUNet: {last_err}")


def tiled_forward(model, x, dev, tile=1024, halo=64):
    """Run the fully-conv model over a large FOV in overlapping tiles."""
    H, W = x.shape
    out = np.zeros((2, H, W), np.float32)
    for y0 in range(0, H, tile):
        for x0 in range(0, W, tile):
            ya, yb = max(0, y0 - halo), min(H, y0 + tile + halo)
            xa, xb = max(0, x0 - halo), min(W, x0 + tile + halo)
            patch = torch.from_numpy(x[ya:yb, xa:xb])[None, None, None].to(dev)
            with torch.no_grad():
                p = model(patch).float().cpu().numpy()[0, :, 0]
            yi0, yi1 = y0 - ya, y0 - ya + min(tile, H - y0)
            xi0, xi1 = x0 - xa, x0 - xa + min(tile, W - x0)
            out[:, y0:y0 + (yi1 - yi0), x0:x0 + (xi1 - xi0)] = p[:, yi0:yi1, xi0:xi1]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/work/vscyto2d/VSCyto2D.ckpt")
    ap.add_argument("--data", default="/work/vscyto2d/a549_hoechst_cellmask_test.zarr")
    ap.add_argument("--json", default=None)
    ap.add_argument("--out", default=None, help="montage PNG")
    ap.add_argument("--n", type=int, default=7, help="max FOVs")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(args.ckpt).eval().to(dev)
    print(f"loaded VSCyto2D checkpoint on {dev}")

    plate = open_ome_zarr(args.data, mode="r")
    ch = plate.channel_names
    ci = {c: ch.index(c) for c in ["Phase3D"] + TARGETS}
    per = {c: [] for c in TARGETS}
    floor = {c: [] for c in TARGETS}
    rows = []
    for name, pos in list(plate.positions())[: args.n]:
        arr = np.asarray(pos["0"])[0]                       # (c, z, y, x)
        phase = arr[ci["Phase3D"], 0].astype(np.float32)
        st = pos.zattrs["normalization"]["Phase3D"]["fov_statistics"]
        x = (phase - st["median"]) / (st["iqr"] + 1e-8)
        pred = tiled_forward(model, x, dev)
        truth = {c: arr[ci[c], 0].astype(np.float32) for c in TARGETS}
        for i, c in enumerate(TARGETS):
            per[c].append(pcc(pred[i], truth[c]))
        rows.append((name, phase, pred, truth))
        print(f"  FOV {name}: " + ", ".join(f"{c} PCC {per[c][-1]:+.3f}" for c in TARGETS))

    # mean-image floor across FOVs (same baseline concept as the BSCCM eval)
    for c in TARGETS:
        mean_img = np.mean([r[3][c] for r in rows], axis=0)
        floor[c] = [pcc(mean_img, r[3][c]) for r in rows]

    print("\nVSCyto2D oracle — predicted vs experimental fluorescence:")
    for c in TARGETS:
        print(f"  {c:6s}  PCC {np.mean(per[c]):+.3f}   (mean-image floor {np.mean(floor[c]):+.3f})")
    overall = float(np.mean([np.mean(per[c]) for c in TARGETS]))
    fl = float(np.mean([np.mean(floor[c]) for c in TARGETS]))
    print(f"  OVERALL PCC {overall:+.3f}  (floor {fl:+.3f}; paper nuclei PCC ~0.71)")

    if args.json:
        json.dump({"overall_pcc": round(overall, 4), "floor": round(fl, 4),
                   "beats_floor": bool(overall > fl), "n_fov": len(rows),
                   "per_channel": {c: {"pcc": round(float(np.mean(per[c])), 4),
                                       "floor": round(float(np.mean(floor[c])), 4)} for c in TARGETS}},
                  open(args.json, "w"), indent=2)
        print("wrote", args.json)

    if args.out:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        n = min(3, len(rows))
        fig, ax = plt.subplots(n, 5, figsize=(16, 3.3 * n))
        ax = np.atleast_2d(ax)
        for r in range(n):
            name, phase, pred, truth = rows[r]
            cols = [(phase, "Phase3D", "gray"),
                    (pred[0], "pred Nucl", "magma"), (truth["Nucl"], "true Nucl", "magma"),
                    (pred[1], "pred Mem", "viridis"), (truth["Mem"], "true Mem", "viridis")]
            for cc, (im, t, cm) in enumerate(cols):
                ax[r, cc].imshow(im, cmap=cm); ax[r, cc].axis("off")
                if r == 0:
                    ax[r, cc].set_title(t, fontsize=10)
        fig.suptitle("VSCyto2D oracle: published checkpoint on official A549 test set", fontsize=12)
        fig.tight_layout(); fig.savefig(args.out, dpi=100)
        print("wrote", args.out)


if __name__ == "__main__":
    main()
