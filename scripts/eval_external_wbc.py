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


def img_tensor(path, size=64, color_norm="absolute"):
    """Resized (3, size, size) float image.

    color_norm="absolute": keep absolute colour, center to [-1,1]. Stain hue is
      discriminative WITHIN a dataset (eosinophil granules, etc.) — but makes the
      model fragile to a white-balance/stain shift across microscopes.
    color_norm="standardize": per-image, per-channel z-score. Removes a global
      colour cast (white-balance invariance) at the cost of absolute-hue cues —
      the fix for cross-microscope domain shift (e.g. Raabin Test-B)."""
    from PIL import Image

    im = np.asarray(Image.open(path).convert("RGB").resize((size, size)), np.float32) / 255.0
    im = np.transpose(im, (2, 0, 1))                         # HWC -> CHW
    if color_norm == "standardize":
        for c in range(3):
            im[c] = (im[c] - im[c].mean()) / (im[c].std() + 1e-6)
        return im
    return im * 2.0 - 1.0                                    # [-1,1]


def _descend(root):
    """Descend through single-subdir wrappers (e.g. Train/Train/) to the class dirs."""
    while True:
        subs = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
        if len(subs) == 1 and not glob.glob(os.path.join(root, subs[0], "*." + "jpg")):
            # a lone wrapper dir with no images directly under it -> descend
            nxt = os.path.join(root, subs[0])
            if any(os.path.isdir(os.path.join(nxt, d)) for d in os.listdir(nxt)):
                root = nxt; continue
        return root


def load_folder(root, n_per_class, seed=0, classes=None):
    """root/<class>/*.img -> (paths, labels, class_names), balanced-capped.

    If `classes` (a canonical name list) is given, labels map into it — so a
    train folder and separate test folders share a consistent label space.
    """
    root = _descend(root)
    dirs = sorted(
        d for d in os.listdir(root)
        if os.path.isdir(os.path.join(root, d))
        and glob.glob(os.path.join(root, d, "**", "*"), recursive=True)
    )
    canon_names = classes if classes is not None else sorted(set(canon(d) for d in dirs))
    rng = np.random.default_rng(seed)
    paths, labels = [], []
    for d in dirs:
        cn = canon(d)
        if cn not in canon_names:
            continue
        files = [f for f in glob.glob(os.path.join(root, d, "**", "*"), recursive=True)
                 if f.lower().endswith(IMG_EXT)]
        if len(files) > n_per_class:
            files = list(rng.permutation(files)[:n_per_class])
        paths += files
        labels += [canon_names.index(cn)] * len(files)
    return paths, np.asarray(labels, int), list(canon_names)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="root folder; subdirs are class names")
    ap.add_argument("--n-per-class", type=int, default=2000)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--model", choices=["mlp", "cnn"], default="mlp",
                    help="mlp = compact descriptors (fast, CPU); cnn = Equinox conv net")
    ap.add_argument("--img-size", type=int, default=64)
    ap.add_argument("--color-norm", choices=["absolute", "standardize"], default="absolute",
                    help="standardize = white-balance-invariant (per-image per-channel z-score); "
                         "use for cross-microscope domain shift")
    ap.add_argument("--augment", action="store_true",
                    help="CNN only: train-time colour-jitter + flip augmentation for domain robustness")
    ap.add_argument("--test-data", nargs="+", default=None,
                    help="separate test folders (e.g. Raabin TestA TestB) for a "
                         "train->test domain-shift protocol; trains on --data")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    def build_X(paths):
        if args.model == "cnn":
            return np.stack([img_tensor(p, args.img_size, args.color_norm) for p in paths])
        return np.stack([img_features(p) for p in paths])

    def fit(Xtr, ytr, nc):
        if args.model == "cnn":
            m, _ = ph.train_cnn_classifier(Xtr, ytr, n_classes=nc, steps=args.steps, augment=args.augment)
            return ("cnn", m)
        m, norm = ph.train_classifier(Xtr, ytr, n_classes=nc, steps=args.steps)
        return ("mlp", m, norm)

    def infer(model, X):
        return ph.predict_cnn(model[1], X) if model[0] == "cnn" else ph.predict_classes(model[1], model[2], X)

    def report(name, y_true, pred, names, train_labels):
        acc = float(np.mean(pred == y_true))
        maj = float(np.mean(y_true == np.bincount(train_labels).argmax()))
        macro = float(np.mean([np.mean(pred[y_true == c] == c) for c in range(len(names)) if (y_true == c).any()]))
        print(f"\n{name} (n={len(y_true)}):  accuracy {acc:.3f}  macro-recall {macro:.3f}  (majority {maj:.3f})")
        per = {}
        for c, nm in enumerate(names):
            m = y_true == c
            r = float(np.mean(pred[m] == c)) if m.any() else float("nan")
            per[nm] = round(r, 4)
            print(f"    {nm:14s} {r:.3f}  (n={int(m.sum())})")
        return {"set": name, "accuracy": round(acc, 4), "macro_recall": round(macro, 4),
                "majority": round(maj, 4), "per_class_recall": per}

    paths, y, names = load_folder(args.data, args.n_per_class)
    print(f"TRAIN {len(paths)} images | {len(names)} classes: {names}")
    print(f"class balance {np.bincount(y, minlength=len(names))}")
    print(f"building inputs ({args.model}) ...")
    X = build_X(paths)

    results = []
    if args.test_data:                       # train->test domain-shift protocol
        model = fit(X, y, len(names))
        for td in args.test_data:
            tp, ty, _ = load_folder(td, args.n_per_class, classes=names)
            results.append(report(os.path.basename(td.rstrip("/")), ty, infer(model, build_X(tp)), names, y))
    else:                                    # single-folder internal split
        rng = np.random.default_rng(1)
        tr, va = [], []
        for c in range(len(names)):
            idx = np.where(y == c)[0]; rng.shuffle(idx)
            k = int(len(idx) * args.val_frac)
            va += list(idx[:k]); tr += list(idx[k:])
        tr, va = np.asarray(tr), np.asarray(va)
        model = fit(X[tr], y[tr], len(names))
        results.append(report("held-out", y[va], infer(model, X[va]), names, y[tr]))

    if args.json:
        import json
        json.dump({"classes": names, "results": results}, open(args.json, "w"), indent=2)
        print("wrote", args.json)


if __name__ == "__main__":
    main()
