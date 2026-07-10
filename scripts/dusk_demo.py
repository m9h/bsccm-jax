"""DUSK demo — differentiable-ODE sensor-kinetics deconvolution (diffrax tier).

Shows DUSK's signature result: the SAME underlying [Ca] waveform is recovered
regardless of which sensor (slow jgcamp8s / fast jgcamp8m) imaged it, whereas the
equilibrium assumption leaves each sensor's kinetic blur in place — a different,
sensor-dependent distortion. Physics-based learning on the Kidger stack: diffrax
solves the binding ODE, an equinox deep prior + optax fit it to the measurement.

    PYTHONPATH=src python scripts/dusk_demo.py --out dusk_demo.png
"""

import argparse

import jax.numpy as jnp
import numpy as np

from bsccm_jax import kinetics as K


def corr(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return float(np.corrcoef(a, b)[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="dusk_demo.png")
    ap.add_argument("--sensors", nargs="+", default=["jgcamp8s", "jgcamp8m"])
    args = ap.parse_args()

    ts = jnp.linspace(0.0, 5.0, 400)
    c_true = K.calcium_transients(ts)
    ct = np.asarray(c_true)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(len(args.sensors), 2, figsize=(13, 4.2 * len(args.sensors)))
    ax = np.atleast_2d(ax)
    tn = np.asarray(ts)
    for r, sensor in enumerate(args.sensors):
        kf, kb = K.SENSORS[sensor]
        f = K.simulate(c_true, ts, kf, kb, seed=1)
        c_eq = np.asarray(K.recover_equilibrium(f, kf, kb))
        c_dusk = np.asarray(K.recover_dusk(f, ts, kf, kb))
        print(f"{sensor}: Kd {K.kd(kf,kb):.0f}nM  corr equilibrium {corr(ct,c_eq):.3f}  "
              f"DUSK {corr(ct,c_dusk):.3f}")

        a0 = ax[r, 0]
        a0.plot(tn, np.asarray(f), color="0.5", lw=1)
        a0.set_title(f"{sensor}: measured fluorescence (Kd {K.kd(kf,kb):.0f} nM)", fontsize=10)
        a0.set_xlabel("time (s)"); a0.set_ylabel("F")

        a1 = ax[r, 1]
        a1.plot(tn, ct, "k", lw=2, label="true [Ca]")
        a1.plot(tn, c_eq * ct.max() / (c_eq.max() + 1e-9), "--", color="tab:orange",
                lw=1.3, label=f"equilibrium (r={corr(ct,c_eq):.2f})")
        a1.plot(tn, c_dusk, color="tab:blue", lw=1.5, label=f"DUSK (r={corr(ct,c_dusk):.2f})")
        a1.set_title(f"{sensor}: recovered [Ca] — DUSK removes the sensor kinetics", fontsize=10)
        a1.set_xlabel("time (s)"); a1.set_ylabel("[Ca] (nM)"); a1.legend(fontsize=8)

    fig.suptitle("DUSK (diffrax): the same [Ca] waveform is recovered from slow & fast sensors; "
                 "equilibrium keeps each sensor's kinetic blur", fontsize=12)
    fig.tight_layout(); fig.savefig(args.out, dpi=110)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
