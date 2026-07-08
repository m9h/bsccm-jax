"""Train + evaluate Tier 1 phenotyping on the full BSCCM dataset.

Runs the classification (WBC type) and regression (surface markers) heads with a
held-out split, reporting accuracy / correlation against the published benchmarks
(~88-91% classification, ~0.72 CD16 regression). Requires the extracted full
dataset (the tiny subset has too few labeled cells).

    uv run python scripts/train_phenotype.py --data /mnt/t9/bsccm_full/BSCCM --n 4000
"""

import argparse

import numpy as np
from bsccm import BSCCM

from bsccm_jax import phenotype as ph


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/mnt/t9/bsccm_full/BSCCM")
    ap.add_argument("--n", type=int, default=4000, help="max labeled cells to use")
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--steps", type=int, default=2000)
    args = ap.parse_args()

    data = BSCCM(args.data, cache_index=True)
    gi, labels = data.get_cell_type_classification_data()
    gi = np.asarray(gi).astype(int)
    labels = np.asarray(labels).astype(int)
    if len(gi) > args.n:
        sel = np.random.default_rng(0).permutation(len(gi))[: args.n]
        gi, labels = gi[sel], labels[sel]
    print(f"{len(gi)} labeled cells | class balance {np.bincount(labels)}")

    print("extracting label-free features ...")
    X = ph.extract_features(data, gi)

    # held-out split
    rng = np.random.default_rng(1)
    perm = rng.permutation(len(gi))
    nval = int(len(gi) * args.val_frac)
    va, tr = perm[:nval], perm[nval:]

    # --- classification ---
    model, norm = ph.train_classifier(X[tr], labels[tr], n_classes=int(labels.max() + 1),
                                      steps=args.steps)
    pred = ph.predict_classes(model, norm, X[va])
    acc = float(np.mean(pred == labels[va]))
    # majority-class baseline
    maj = float(np.mean(labels[va] == np.bincount(labels[tr]).argmax()))
    print(f"\nCLASSIFICATION (held-out {nval} cells):")
    print(f"  accuracy         {acc:.3f}   (majority baseline {maj:.3f}; benchmark ~0.88-0.91)")

    # --- regression (surface markers), if available ---
    try:
        sm = data.get_surface_marker_data(gi)
        names = list(sm[0]); Y = np.asarray(sm[1], np.float32)   # (N, 9) marker abundances
        reg, rn = ph.train_regressor(X[tr], Y[tr], steps=args.steps)
        import jax, jax.numpy as jnp
        mu, sd, ym, ys = rn
        pr = np.asarray(jax.vmap(reg)((jnp.asarray(X[va]) - mu) / sd)) * np.asarray(ys) + np.asarray(ym)
        cors = [np.corrcoef(pr[:, k], Y[va][:, k])[0, 1] for k in range(Y.shape[1])]
        print(f"\nREGRESSION (surface markers, held-out):")
        print(f"  mean Pearson     {np.nanmean(cors):.3f}")
        cd16 = [i for i, n in enumerate(names) if "CD16" in n]
        if cd16:
            print(f"  CD16 Pearson     {cors[cd16[0]]:.3f}   (benchmark ~0.72)")
    except Exception as e:
        import traceback; traceback.print_exc()
        print(f"\nREGRESSION skipped: {type(e).__name__}: {str(e)[:100]}")


if __name__ == "__main__":
    main()
