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
    # Surface-marker regression: each cell is stained for ONE antibody, so we
    # regress a given marker on THAT marker's stained cells (the benchmark setup).
    try:
        marker = "CD16"
        m_idx = np.asarray([int(i) for i in data.get_indices(antibodies=marker)])
        if len(m_idx) > args.n:
            m_idx = m_idx[np.random.default_rng(3).permutation(len(m_idx))[:args.n]]
        print(f"\nREGRESSION {marker}: {len(m_idx)} stained cells; extracting features ...")
        Xm = ph.extract_features(data, m_idx)
        sm = data.get_surface_marker_data([int(i) for i in m_idx])
        names = list(sm[0]); Ym = np.asarray(sm[1], np.float32)
        col = next(k for k, nm in enumerate(names) if marker in nm)
        y = Ym[:, col]; keep = ~np.isnan(y)
        Xm, y = Xm[keep], y[keep]
        rp = np.random.default_rng(2).permutation(len(y)); nv = int(len(y) * 0.2)
        vv, tt = rp[:nv], rp[nv:]
        reg, rn = ph.train_regressor(Xm[tt], y[tt], steps=args.steps)
        import jax, jax.numpy as jnp
        mu, sd, ym, ys = rn
        pr = np.asarray(jax.vmap(reg)((jnp.asarray(Xm[vv]) - mu) / sd))[:, 0] * float(ys[0]) + float(ym[0])
        r = float(np.corrcoef(pr, y[vv])[0, 1])
        print(f"  CD16 Pearson  {r:.3f}  (held-out {nv}; benchmark ~0.72)")
    except Exception as e:
        import traceback; traceback.print_exc()
        print(f"\nREGRESSION skipped: {type(e).__name__}: {str(e)[:100]}")


if __name__ == "__main__":
    main()
