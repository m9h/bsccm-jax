"""DUSK on a REAL jGCaMP8s recording with ground-truth spikes (CASCADE DS32).

The exact cell DUSK references — CAttached_jGCaMP8s_479571_2 — imaged with
jGCaMP8s (the sensor our kinetic constants are calibrated to) alongside cell-
attached electrophysiology, so we have the true spike times. Question: does
inverting the sensor ODE recover a [Ca] that tracks real spikes better than the
equilibrium assumption, on genuine (not simulated) fluorescence?

Get the data (CASCADE, GPL-3.0 — cite Rupprecht et al. 2025; not redistributed here):
    wget -P data https://raw.githubusercontent.com/HelmchenLabSoftware/Cascade/master/\
Ground_truth/DS32-GCaMP8s-m-V1/CAttached_jGCaMP8s_479571_2_mini.mat
    mv data/CAttached_jGCaMP8s_479571_2_mini.mat data/jgcamp8s_479571_2.mat

    PYTHONPATH=src python scripts/dusk_real.py --mat data/jgcamp8s_479571_2.mat \
        --t0 65 --dur 22 --out dusk_real.png

Data: fluo_time [s], fluo_mean [dF/F], events_AP [0.1 ms ticks -> s].

Honest finding: on real data DUSK yields a clean, denoised calcium estimate that
separates closely-spaced bursts better than the raw fluorescence, but does NOT
beat the (calibrated) fluorescence on spike-correlation (0.82 vs 0.94) — that
metric rewards the sensor-blurred waveform, and fluorescence already tracks
spikes. Deconvolution's real value isn't captured by spike-timing, and there is
no ground-truth *calcium* to score against; the clean quantitative validation of
DUSK's sensor-invariant recovery stays the synthetic experiment (dusk_demo.py).
"""

import argparse

import numpy as np
import scipy.io as sio

from bsccm_jax import kinetics as K


def load(mat):
    m = sio.loadmat(mat, squeeze_me=True, struct_as_record=False)
    r = np.atleast_1d(m["CAttached"])[0]
    t = np.asarray(r.fluo_time, float).ravel()
    dff = np.asarray(r.fluo_mean, float).ravel()
    sp = np.asarray(r.events_AP, float).ravel() * 1e-4
    return t, dff, sp


def spike_calcium(ts, spikes, tau=0.4):
    """Ground-truth-derived calcium proxy: each spike -> exp-decay kernel."""
    ca = np.zeros_like(ts)
    for s in spikes:
        d = ts - s
        ca += np.where(d >= 0, np.exp(-d / tau), 0.0)
    return ca


def corr(a, b):
    a, b = np.asarray(a), np.asarray(b)
    a = (a - a.mean()) / (a.std() + 1e-9); b = (b - b.mean()) / (b.std() + 1e-9)
    return float((a * b).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mat", default="data/jgcamp8s_479571_2.mat")
    ap.add_argument("--t0", type=float, default=65.0)
    ap.add_argument("--dur", type=float, default=22.0)
    ap.add_argument("--sensor", default="jgcamp8s")
    ap.add_argument("--out", default="dusk_real.png")
    args = ap.parse_args()

    t, dff, sp = load(args.mat)
    kf, kb = K.SENSORS[args.sensor]
    sel = (t >= args.t0) & (t < args.t0 + args.dur)
    ts, f = t[sel], dff[sel]
    spikes = sp[(sp >= args.t0) & (sp < args.t0 + args.dur)]
    print(f"{args.sensor}: window {args.t0}-{args.t0+args.dur}s, {len(ts)} samples, {len(spikes)} spikes")

    c_dusk, qe, bg = K.recover_dusk_calibrated(f, ts, kf, kb)
    c_dusk = np.asarray(c_dusk)
    c_eq = np.asarray(K.recover_equilibrium(f, kf, kb, qe=qe, bg=bg))
    ca_gt = spike_calcium(ts, spikes)                       # spike-derived calcium proxy

    r_dusk, r_eq = corr(c_dusk, ca_gt), corr(c_eq, ca_gt)
    print(f"corr with spike-derived calcium:  DUSK {r_dusk:+.3f}   equilibrium {r_eq:+.3f}   "
          f"(fitted qe {qe:.2f}, bg {bg:+.2f})")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(3, 1, figsize=(13, 8), sharex=True)
    ax[0].plot(ts, f, color="0.4", lw=0.8); ax[0].set_ylabel("dF/F")
    ax[0].set_title(f"real jGCaMP8s fluorescence (cell 479571) + {len(spikes)} ground-truth spikes")
    for s in spikes:
        ax[0].axvline(s, color="tab:red", alpha=0.3, lw=0.6)
    ax[1].plot(ts, c_eq, color="tab:orange", lw=1)
    ax[1].set_ylabel("[Ca] (a.u.)"); ax[1].set_title(f"equilibrium recovery (r={r_eq:+.2f} vs spikes)")
    for s in spikes:
        ax[1].axvline(s, color="tab:red", alpha=0.3, lw=0.6)
    ax[2].plot(ts, c_dusk, color="tab:blue", lw=1.2)
    ax[2].set_ylabel("[Ca] (nM)"); ax[2].set_xlabel("time (s)")
    ax[2].set_title(f"DUSK recovery — sensor ODE inverted (r={r_dusk:+.2f} vs spikes)")
    for s in spikes:
        ax[2].axvline(s, color="tab:red", alpha=0.3, lw=0.6)
    fig.suptitle("DUSK on real jGCaMP8s (CASCADE ground truth): recovered [Ca] vs true spikes", fontsize=12)
    fig.tight_layout(); fig.savefig(args.out, dpi=110)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
