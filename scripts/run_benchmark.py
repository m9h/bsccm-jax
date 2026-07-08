"""Full-dataset BSCCM benchmark — one command, all the numbers.

Runs every task the platform covers on a given BSCCM root and prints a
consolidated results table with the published reference points:

  * cell-type classification   (label-free -> WBC type)   ref ~0.88-0.91 acc
  * surface-marker regression  (label-free -> abundance)   ref ~0.72 Pearson
  * reconstruction validation  (DPC / FPM / NeuPh, synthetic self-tests)

Virtual staining (label-free -> fluorescence) is trained separately (VisCy /
Modal ``train_staining``); its held-out corr/SSIM-vs-baseline is folded in if a
metrics file is passed via --staining-metrics.

    uv run python scripts/run_benchmark.py --data /data/BSCCM --n 6000
"""

import argparse
import json
import time

import numpy as np


def bench_classification_regression(data_path, n):
    from bsccm import BSCCM
    from bsccm_jax import phenotype as ph

    data = BSCCM(data_path, cache_index=True)
    gi, labels = data.get_cell_type_classification_data()
    gi = np.asarray(gi).astype(int); labels = np.asarray(labels).astype(int)
    if len(gi) > n:
        sel = np.random.default_rng(0).permutation(len(gi))[:n]
        gi, labels = gi[sel], labels[sel]
    X = ph.extract_features(data, gi)
    rng = np.random.default_rng(1); perm = rng.permutation(len(gi))
    nval = int(len(gi) * 0.2); va, tr = perm[:nval], perm[nval:]

    model, norm = ph.train_classifier(X[tr], labels[tr], n_classes=int(labels.max() + 1), steps=2500)
    acc = float(np.mean(ph.predict_classes(model, norm, X[va]) == labels[va]))
    maj = float(np.mean(labels[va] == np.bincount(labels[tr]).argmax()))
    out = {"classification_accuracy": acc, "classification_majority_baseline": maj,
           "n_labeled": int(len(gi))}

    try:
        import jax, jax.numpy as jnp
        sm = data.get_surface_marker_data(gi)
        Y = np.asarray(sm[1], np.float32)                 # (N, 9) marker abundances
        reg, rn = ph.train_regressor(X[tr], Y[tr], steps=2500)
        mu, sd, ym, ys = rn
        pr = np.asarray(jax.vmap(reg)((jnp.asarray(X[va]) - mu) / sd)) * np.asarray(ys) + np.asarray(ym)
        cors = [np.corrcoef(pr[:, k], Y[va][:, k])[0, 1] for k in range(Y.shape[1])]
        out["regression_mean_pearson"] = float(np.nanmean(cors))
    except Exception as e:
        out["regression_error"] = f"{type(e).__name__}: {str(e)[:80]}"
    return out


def bench_reconstruction():
    from bsccm_jax import dpc, fpm, neuph
    res = {}
    # DPC (analytic) on the phantom
    shape = (96, 96); u, p = dpc.phantom(shape)
    Hp = dpc.dpc_2axis_transfer(shape, wavelength_um=0.532, pixel_size_um=0.2, na=0.5)
    meas = dpc.dpc_apply_phase(Hp, p)
    rec = dpc.reconstruct_dpc_2axis({"Top": meas[0], "Bottom": -meas[0], "Left": meas[1], "Right": -meas[1]})
    # FPM
    hr = fpm.hr_phantom((256, 256)); pup = fpm.circ_pupil((64, 64), 9); sh = fpm.led_grid_shifts(9, 6)
    im = fpm.fpm_forward(hr, sh, pup, (64, 64)); frec = fpm.reconstruct_fpm(im, sh, pup, (256, 256), steps=300)

    def corr(a, b):
        a = np.asarray(a).ravel() - np.mean(a); b = np.asarray(b).ravel() - np.mean(b)
        return float((a * b).sum() / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
    res["fpm_phase_corr"] = corr(np.angle(frec), np.angle(hr))
    # NeuPh generalization
    Hp2 = dpc.dpc_2axis_transfer((64, 64), wavelength_um=0.532, pixel_size_um=0.2, na=0.5)
    d = [(np.asarray(dpc.dpc_apply_phase(Hp2, dpc.phantom((64, 64), seed=s)[1])),
          np.asarray(dpc.phantom((64, 64), seed=s)[1])) for s in range(40)]
    M = np.stack([x[0] for x in d]); P = np.stack([x[1] for x in d])
    _, vc = neuph.train_lcnf(M, P, epochs=40)
    res["neuph_heldout_corr"] = float(vc)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/data/BSCCM")
    ap.add_argument("--n", type=int, default=6000)
    ap.add_argument("--staining-metrics", default=None, help="JSON from eval_vs.py")
    ap.add_argument("--out", default="benchmark_results.json")
    ap.add_argument("--skip-data", action="store_true", help="reconstruction only (no dataset)")
    args = ap.parse_args()

    results = {}
    t0 = time.time()
    if not args.skip_data:
        print("== phenotyping (classification + regression) ==")
        results["phenotyping"] = bench_classification_regression(args.data, args.n)
    print("== reconstruction validation ==")
    results["reconstruction"] = bench_reconstruction()
    if args.staining_metrics:
        results["virtual_staining"] = json.load(open(args.staining_metrics))

    print("\n" + "=" * 62)
    print("  BSCCM FULL-DATASET BENCHMARK".center(62))
    print("=" * 62)
    p = results.get("phenotyping", {})
    if "classification_accuracy" in p:
        print(f"  cell-type classification   {p['classification_accuracy']:.3f}   "
              f"(baseline {p['classification_majority_baseline']:.3f}; ref ~0.88-0.91)")
    if "regression_mean_pearson" in p:
        print(f"  surface-marker regression  {p['regression_mean_pearson']:.3f}   (ref ~0.72)")
    r = results["reconstruction"]
    print(f"  FPM phase recovery         {r['fpm_phase_corr']:.3f}   (super-resolution)")
    print(f"  NeuPh held-out (1-pass)    {r['neuph_heldout_corr']:.3f}   (generalization)")
    vs = results.get("virtual_staining")
    if vs and "overall_corr" in vs:
        print(f"  virtual staining           {vs['overall_corr']:.3f}   (mean-img floor {vs.get('floor','?')})")
    print("=" * 62)
    print(f"  ({time.time()-t0:.0f}s)")
    json.dump(results, open(args.out, "w"), indent=2)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
