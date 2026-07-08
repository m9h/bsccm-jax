"""Evaluate the trained BSCCM virtual-staining model on held-out val cells.

Loads a VSUNet checkpoint, predicts the 6 fluorescence bands from Phase for the
validation FOVs (well A/2), reports per-channel correlation, and renders a
montage: Phase | predicted fluor | true fluor. Runs inside the NGC container.

    /work/vsvenv/bin/python /work/eval_vs.py --ckpt <ckpt> --out /work/vs_eval.png
"""

import argparse

import numpy as np
import torch
from iohub import open_ome_zarr

from cytoland.engine import VSUNet
from viscy_utils.losses import MixedLoss

FLUOR = ["Fluor_690-", "Fluor_627-673", "Fluor_585-625",
         "Fluor_550-570", "Fluor_500-550", "Fluor_426-446"]
MODEL_CONFIG = dict(in_channels=1, out_channels=6, encoder_blocks=[3, 3, 9, 3],
                    dims=[96, 192, 384, 768], decoder_conv_blocks=2,
                    stem_kernel_size=[1, 2, 2], in_stack_depth=1, pretraining=False)


def zscore(a):
    a = np.asarray(a, np.float32)
    return (a - a.mean()) / (a.std() + 1e-6)


def corr(a, b):
    a = zscore(a).ravel(); b = zscore(b).ravel()
    return float((a * b).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", default="/work/data/bsccm_vs.zarr")
    ap.add_argument("--out", default="/work/vs_eval.png")
    ap.add_argument("--json", default=None, help="write metrics as JSON for the benchmark")
    ap.add_argument("--n", type=int, default=5)
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = VSUNet(architecture="fcmae", model_config=MODEL_CONFIG,
                   loss_function=MixedLoss(l1_alpha=1.0))
    sd = torch.load(args.ckpt, weights_only=True, map_location="cpu")["state_dict"]
    model.load_state_dict(sd)
    model.eval().to(dev)

    plate = open_ome_zarr(args.data, mode="r")
    val = [(name, pos) for name, pos in plate.positions() if "/2/" in f"/{name}/"][: args.n]

    try:
        from skimage.metrics import structural_similarity as ssim_fn
        from skimage.metrics import peak_signal_noise_ratio as psnr_fn
    except Exception:
        ssim_fn = psnr_fn = None

    def _norm01(a):
        a = np.asarray(a, float); lo, hi = a.min(), a.max()
        return (a - lo) / (hi - lo + 1e-8)

    rows = []
    per = {c: {"corr": [], "ssim": [], "psnr": [], "base": []} for c in FLUOR}
    true_all = {c: [] for c in FLUOR}
    for name, pos in val:
        arr = np.asarray(pos["0"])[0]
        ch = plate.channel_names
        phase = arr[ch.index("Phase"), 0]
        true = np.stack([arr[ch.index(c), 0] for c in FLUOR])
        x = torch.from_numpy(zscore(phase))[None, None, None].to(dev)
        with torch.no_grad():
            pred = model(x).float().cpu().numpy()[0, :, 0]
        for i, c in enumerate(FLUOR):
            per[c]["corr"].append(corr(pred[i], true[i]))
            true_all[c].append(true[i])
            if ssim_fn is not None:
                pn, tn = _norm01(pred[i]), _norm01(true[i])
                per[c]["ssim"].append(ssim_fn(tn, pn, data_range=1.0))
                per[c]["psnr"].append(psnr_fn(tn, pn, data_range=1.0))
        rows.append((name, phase, pred, true))

    # mean-image baseline (the floor): predict every cell as the dataset-mean image
    for c in FLUOR:
        mean_img = np.mean(true_all[c], axis=0)
        per[c]["base"] = [corr(mean_img, t) for t in true_all[c]]

    print("per-channel metrics (predicted vs true, n=%d):" % len(val))
    print(f"  {'channel':14s} {'corr':>6} {'SSIM':>6} {'PSNR':>6}  {'mean-img corr (floor)':>22}")
    for c in FLUOR:
        m = per[c]
        ssim = np.mean(m["ssim"]) if m["ssim"] else float("nan")
        psnr = np.mean(m["psnr"]) if m["psnr"] else float("nan")
        print(f"  {c:14s} {np.mean(m['corr']):+.3f} {ssim:6.3f} {psnr:6.1f}  {np.mean(m['base']):+.3f}")
    overall = float(np.mean([np.mean(per[c]["corr"]) for c in FLUOR]))
    floor = float(np.mean([np.mean(per[c]["base"]) for c in FLUOR]))
    print(f"  {'OVERALL':14s} {overall:+.3f}  (mean-image floor {floor:+.3f} — model must beat this)")

    if args.json:
        import json
        metrics = {"overall_corr": round(overall, 4), "floor": round(floor, 4),
                   "beats_floor": bool(overall > floor), "n_val": len(val),
                   "per_channel": {c: {"corr": round(float(np.mean(per[c]["corr"])), 4),
                                       "ssim": round(float(np.mean(per[c]["ssim"])), 4) if per[c]["ssim"] else None,
                                       "psnr": round(float(np.mean(per[c]["psnr"])), 2) if per[c]["psnr"] else None,
                                       "floor": round(float(np.mean(per[c]["base"])), 4)} for c in FLUOR}}
        json.dump(metrics, open(args.json, "w"), indent=2)
        print("wrote", args.json)

    # montage: show the most-informative fluor channel (500-550, index 4) per cell
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ci = 4
    fig, ax = plt.subplots(len(rows), 3, figsize=(9, 3 * len(rows)))
    ax = np.atleast_2d(ax)
    for r, (name, phase, pred, true) in enumerate(rows):
        for c, (img, t, cm) in enumerate([(phase, "Phase (input)", "gray"),
                                          (pred[ci], f"predicted {FLUOR[ci]}", "magma"),
                                          (true[ci], f"true {FLUOR[ci]}", "magma")]):
            ax[r, c].imshow(img, cmap=cm); ax[r, c].axis("off")
            if r == 0:
                ax[r, c].set_title(t, fontsize=10)
        ax[r, 1].set_title(f"predicted {FLUOR[ci]}\ncorr={corr(pred[ci], true[ci]):+.2f}", fontsize=9)
    fig.suptitle("BSCCM in-silico labeling: label-free Phase -> fluorescence (Cytoland/VSUNet, GB10)", fontsize=11)
    fig.tight_layout(); fig.savefig(args.out, dpi=110)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
