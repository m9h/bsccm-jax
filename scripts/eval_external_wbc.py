"""Cross-dataset generalization test for the phenotyping classifier.

Our BSCCM white-blood-cell classifier hit ~0.919 on Berkeley label-free data.
This harness asks a different question: does the *method* (compact multi-scale
image descriptors + an Equinox MLP) hold up on INDEPENDENT, externally-collected
stained-blood-smear WBC datasets with their own published accuracies —
Raabin-WBC, PBC-Barcelona, or any Kaggle WBC mirror?

It is deliberately dataset-agnostic: point it at a root folder whose immediate
subdirectories are class names, each holding that class's images (the near-
universal layout for these datasets). It builds features, trains our classifier
on a stratified split, and reports held-out accuracy + per-class recall +
confusion against the majority-class floor.

    PYTHONPATH=src python scripts/eval_external_wbc.py --data /path/to/wbc_root \
        --n-per-class 2000 --json wbc_external.json

The MLP-on-descriptors is the platform's baseline head (fast, CPU-friendly, same
front end as phenotype.py); a small conv net is the documented GPU upgrade if the
descriptor baseline trails the published CNN numbers.
"""

import argparse
import glob
import os

import numpy as np

from bsccm_jax import phenotype as ph

IMG_EXT = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp")
# canonicalize the many spellings these datasets use for the same WBC types
CANON = {
    "neutrophil": "neutrophil", "neutrophils": "neutrophil", "segmented_neutrophil": "neutrophil",
    "band": "neutrophil", "seg": "neutrophil",
    "lymphocyte": "lymphocyte", "lymphocytes": "lymphocyte", "large_lymph": "lymphocyte",
    "monocyte": "monocyte", "monocytes": "monocyte",
    "eosinophil": "eosinophil", "eosinophils": "eosinophil",
    "basophil": "basophil", "basophils": "basophil",
    "ig": "immature_granulocyte", "immature_granulocytes": "immature_granulocyte",
    "erythroblast": "erythroblast", "platelet": "platelet", "platelets": "platelet",
}


def canon(name):
    k = name.strip().lower().replace(" ", "_").replace("-", "_")
    return CANON.get(k, k)


def img_features(path, scales=(24, 12)):
    """Multi-scale low-res RGB descriptor + global color stats.

    Colour is kept ABSOLUTE (only /255), not per-image z-scored — staining
    intensity/hue is itself discriminative for WBC types (eosinophil granules,
    basophil darkness, lymphocyte nucleus:cytoplasm). Per-feature scaling is done
    once at the dataset level inside train_classifier, which is the correct place.
    """
    from PIL import Image

    im = np.asarray(Image.open(path).convert("RGB"), np.float32) / 255.0  # HWC
    feats = []
    for d in scales:
        h, w = im.shape[0] // d, im.shape[1] // d
        if h == 0 or w == 0:
            feats.append(np.zeros(d * d * 3, np.float32))
            continue
        block = im[: h * d, : w * d].reshape(d, h, d, w, 3).mean((1, 3))  # (d,d,3)
        feats.append(block.ravel())
    flat = im.reshape(-1, 3)
    feats.append(flat.mean(0))
    feats.append(flat.std(0))
    return np.concatenate(feats).astype(np.float32)


def img_tensor(path, size=64):
    """Resized (3, size, size) float image in [-1,1], colour preserved.

    Deliberately NOT per-image standardized: stain hue/intensity is
    discriminative for WBC types, so we keep absolute colour and just center to
    [-1,1] (the conv net learns its own per-channel scaling from the batch)."""
    from PIL import Image

    im = np.asarray(Image.open(path).convert("RGB").resize((size, size)), np.float32) / 255.0
    return np.transpose(im, (2, 0, 1)) * 2.0 - 1.0           # HWC->CHW, [-1,1]


def load_folder(root, n_per_class, seed=0):
    """root/<class>/*.img -> (paths, labels, class_names), balanced-capped."""
    classes = sorted(
        d for d in os.listdir(root)
        if os.path.isdir(os.path.join(root, d))
        and glob.glob(os.path.join(root, d, "**", "*"), recursive=True)
    )
    # merge classes that canonicalize to the same WBC type
    canon_names = sorted(set(canon(c) for c in classes))
    cls_to_idx = {c: canon_names.index(canon(c)) for c in classes}
    rng = np.random.default_rng(seed)
    paths, labels = [], []
    for c in classes:
        files = [f for f in glob.glob(os.path.join(root, c, "**", "*"), recursive=True)
                 if f.lower().endswith(IMG_EXT)]
        if len(files) > n_per_class:
            files = list(rng.permutation(files)[:n_per_class])
        paths += files
        labels += [cls_to_idx[c]] * len(files)
    return paths, np.asarray(labels, int), canon_names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="root folder; subdirs are class names")
    ap.add_argument("--n-per-class", type=int, default=2000)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--model", choices=["mlp", "cnn"], default="mlp",
                    help="mlp = compact descriptors (fast, CPU); cnn = Equinox conv net")
    ap.add_argument("--img-size", type=int, default=64)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    paths, y, names = load_folder(args.data, args.n_per_class)
    print(f"{len(paths)} images | {len(names)} classes: {names}")
    print(f"class balance {np.bincount(y, minlength=len(names))}")

    print(f"loading inputs ({args.model}) ...")
    if args.model == "cnn":
        X = np.stack([img_tensor(p, args.img_size) for p in paths])
    else:
        X = np.stack([img_features(p) for p in paths])

    # stratified held-out split
    rng = np.random.default_rng(1)
    tr, va = [], []
    for c in range(len(names)):
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        k = int(len(idx) * args.val_frac)
        va += list(idx[:k]); tr += list(idx[k:])
    tr, va = np.asarray(tr), np.asarray(va)

    if args.model == "cnn":
        model, _ = ph.train_cnn_classifier(X[tr], y[tr], n_classes=len(names), steps=args.steps)
        pred = ph.predict_cnn(model, X[va])
    else:
        model, norm = ph.train_classifier(X[tr], y[tr], n_classes=len(names), steps=args.steps)
        pred = ph.predict_classes(model, norm, X[va])
    acc = float(np.mean(pred == y[va]))
    maj = float(np.mean(y[va] == np.bincount(y[tr]).argmax()))

    # per-class recall + confusion
    print(f"\nCROSS-DATASET WBC (held-out {len(va)} imgs):")
    print(f"  accuracy   {acc:.3f}   (majority floor {maj:.3f})")
    print("  per-class recall:")
    per = {}
    for c, nm in enumerate(names):
        m = y[va] == c
        r = float(np.mean(pred[m] == c)) if m.any() else float("nan")
        per[nm] = round(r, 4)
        print(f"    {nm:22s} {r:.3f}  (n={int(m.sum())})")

    if args.json:
        import json
        out = {"accuracy": round(acc, 4), "majority_floor": round(maj, 4),
               "n_val": len(va), "classes": names, "per_class_recall": per}
        json.dump(out, open(args.json, "w"), indent=2)
        print("wrote", args.json)


if __name__ == "__main__":
    main()
