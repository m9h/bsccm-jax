"""IDEAL teaching module — imaging as an information channel, on real BSCCM cells.

A hands-on demonstration of the core idea behind Information-Driven Design of
Imaging Systems (Pinkard & Waller, NeurIPS 2025): a measurement is good not
because it looks nice, but because it carries **information** about the object.
Here the "object" is a cell's fluorescence marker level, and the "measurements"
are BSCCM's label-free channels. Students see that the measurements carrying more
information about the marker are exactly the ones from which it can be predicted.

Percent-cell script (run interactively, or `python examples/ideal_teaching.py`).
Browser/zero-hardware friendly: needs only the BSCCM tiny subset + this repo's env.
For rigorous mutual-information estimation, swap the Gaussian proxy below for
Waller-Lab/EncodingInformation (the IDEAL codebase, JAX).
"""

# %% imports
import numpy as np
from bsccm import BSCCM

# %% [markdown]
# ## 1. The setup
# Each BSCCM cell is imaged under many *label-free* contrasts (brightfield, a
# darkfield ring, differential phase contrast) and also has *fluorescence*
# measurements of surface-protein markers. Label-free imaging is cheap and
# non-destructive; fluorescence needs stains. The question IDEAL asks:
# **which label-free measurements carry the most information about the marker?**

# %% load cells: label-free channel groups + a fluorescence target
data = BSCCM("data/BSCCM-tiny", cache_index=True)
idx = [int(i) for i in data.get_indices()[:400]]

GROUPS = {
    "Brightfield":  ["Brightfield"],
    "DPC (4)":      ["DPC_Top", "DPC_Bottom", "DPC_Left", "DPC_Right"],
    "Darkfield ring": ["DF_50", "DF_60", "DF_70", "DF_80", "DF_90"],
    "all label-free": None,   # filled below
}
GROUPS["all label-free"] = GROUPS["Brightfield"] + GROUPS["DPC (4)"] + GROUPS["Darkfield ring"]
TARGET = "Fluor_500-550"      # the marker we try to recover (well-predicted band)


D = 8  # work at low resolution so a simple linear model suffices for teaching


def _down(im):
    im = np.asarray(im, np.float32)
    h = im.shape[0] // D
    return (im[:h * D, :h * D].reshape(D, h, D, h).mean((1, 3))).ravel()  # (D*D,)


def _zc(im):  # per-image z-score then downsample (matches the training normalization)
    im = np.asarray(im, np.float32)
    return _down((im - im.mean()) / (im.std() + 1e-6))


def feats(indices, channels):
    """Spatial features: each channel downsampled to DxD, concatenated."""
    return np.asarray([np.concatenate([_zc(data.read_image(i, c, copy=True)) for c in channels])
                       for i in indices])


def target(indices):
    """The fluorescence marker as a low-res image (the thing to recover)."""
    return np.asarray([_zc(data.read_image(i, TARGET, copy=True)) for i in indices])


# %% [markdown]
# ## 2. Two numbers per measurement set
# * **Information** — a Gaussian estimate of the mutual information I(measurements;
#   marker): how much knowing the measurements reduces uncertainty about the marker.
# * **Recoverability** — held-out correlation of a simple linear predictor. This is
#   what you can actually *do* with the measurements.
# IDEAL's thesis: the first predicts the second, *without ever training a decoder*.


def gaussian_mi(X, y):
    """Mean per-pixel MI(X; target_pixel) under a joint-Gaussian assumption."""
    Xn = (X - X.mean(0)) / (X.std(0) + 1e-8)
    A = np.c_[Xn, np.ones(len(Xn))]
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)          # (nfeat+1, P)
    resid = y - A @ beta
    mi = 0.5 * np.log((y.var(0) + 1e-8) / (resid.var(0) + 1e-8))
    return float(np.mean(np.clip(mi, 0, None)))


def recoverability(X, y, k=5):
    """k-fold held-out correlation of a ridge predictor of the low-res image."""
    n = len(X); fold = n // k; cors = []
    for f in range(k):
        te = np.zeros(n, bool); te[f * fold:(f + 1) * fold] = True
        Xtr, Xte = X[~te], X[te]
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
        Atr = np.c_[(Xtr - mu) / sd, np.ones((~te).sum())]
        Ate = np.c_[(Xte - mu) / sd, np.ones(te.sum())]
        beta = np.linalg.solve(Atr.T @ Atr + 1e-1 * np.eye(Atr.shape[1]), Atr.T @ y[~te])
        p, t = Ate @ beta, y[te]
        a, b = (p - p.mean()).ravel(), (t - t.mean()).ravel()
        cors.append((a * b).sum() / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))
    return float(np.mean(cors))


# %% compute for each measurement group
y = target(idx)
results = {}
for name, chans in GROUPS.items():
    X = feats(idx, chans)
    results[name] = (gaussian_mi(X, y), recoverability(X, y))
    print(f"{name:16s}  info={results[name][0]:+.3f} nats   recoverability(corr)={results[name][1]:+.3f}")

# %% [markdown]
# ## 3. The lesson
# Plot information vs recoverability — they rise together. Richer coded
# measurements (the DPC set, the darkfield ring, all combined) carry more
# information about the marker *and* let you predict it better; brightfield alone
# carries the least of both. That is the whole argument for computational
# imaging, made measurable — and the basis for *designing* which measurements to
# take (IDEAL) before ever building a reconstructor.

# %% plot
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(6.5, 5))
for name, (mi, rec) in results.items():
    ax.scatter(mi, rec, s=90)
    ax.annotate(name, (mi, rec), textcoords="offset points", xytext=(8, 4), fontsize=9)
ax.set_xlabel("information about the marker  I(measurements; marker)  [nats]")
ax.set_ylabel("recoverability  (held-out prediction correlation)")
ax.set_title("IDEAL on BSCCM: information predicts what you can recover\n"
             f"(target: {TARGET}, n={len(idx)} cells)")
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig("ideal_teaching.png", dpi=110)
print("wrote ideal_teaching.png")
