"""Label-free single-cell phenotyping — Tier 1 of the BSCCM benchmark.

Two heads off the same label-free front end:
  * classification — predict WBC type (Lymphocyte / Granulocyte / Monocyte),
    the task with published benchmarks (~88-91% accuracy).
  * regression — predict surface-marker abundance (e.g. CD16), ~0.72 Pearson.

Kept JAX/Equinox + Optax, consistent with the platform. Features are compact
low-res multi-channel descriptors (fast on CPU); swap for a conv net on the GB10
for the full-dataset run. Requires the full BSCCM (the tiny subset has only ~28
labeled cells); the code is dataset-agnostic and runs on either.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

LABEL_FREE = ["Brightfield", "DPC_Top", "DPC_Bottom", "DPC_Left", "DPC_Right",
              "DF_50", "DF_60", "DF_70", "DF_80", "DF_90"]
D = 12  # downsample resolution for features


def _feat(im):
    im = np.asarray(im, np.float32)
    im = (im - im.mean()) / (im.std() + 1e-6)
    h = im.shape[0] // D
    return im[:h * D, :h * D].reshape(D, h, D, h).mean((1, 3)).ravel()


def extract_features(data, indices, channels=LABEL_FREE):
    """(N, D*D*len(channels)) label-free feature matrix."""
    return np.stack([
        np.concatenate([_feat(data.read_image(int(i), c, copy=True)) for c in channels])
        for i in indices
    ]).astype(np.float32)


class MLP(eqx.Module):
    layers: list

    def __init__(self, in_dim, out_dim, width=256, depth=3, *, key):
        keys = jax.random.split(key, depth + 1)
        dims = [in_dim] + [width] * (depth - 1) + [out_dim]
        self.layers = [eqx.nn.Linear(dims[i], dims[i + 1], key=keys[i]) for i in range(depth)]

    def __call__(self, x):
        for lyr in self.layers[:-1]:
            x = jax.nn.gelu(lyr(x))
        return self.layers[-1](x)


def _train(model, X, y, loss_fn, steps=800, lr=1e-3, batch=64, seed=0):
    opt = optax.adamw(lr)
    opt_state = opt.init(eqx.filter(model, eqx.is_array))

    @eqx.filter_jit
    def step(model, opt_state, xb, yb):
        loss, grads = eqx.filter_value_and_grad(lambda m: loss_fn(m, xb, yb))(model)
        updates, opt_state = opt.update(grads, opt_state, eqx.filter(model, eqx.is_array))
        return eqx.apply_updates(model, updates), opt_state, loss

    key = jax.random.PRNGKey(seed)
    n = len(X)
    for i in range(steps):
        key, sk = jax.random.split(key)
        idx = jax.random.randint(sk, (min(batch, n),), 0, n)
        model, opt_state, _ = step(model, opt_state, X[idx], y[idx])
    return model


def train_classifier(X, y, n_classes, **kw):
    Xj, yj = jnp.asarray(X), jnp.asarray(y)
    mu, sd = Xj.mean(0), Xj.std(0) + 1e-6
    Xn = (Xj - mu) / sd
    model = MLP(X.shape[1], n_classes, key=jax.random.PRNGKey(0))

    def loss_fn(m, xb, yb):
        logits = jax.vmap(m)(xb)
        return optax.softmax_cross_entropy_with_integer_labels(logits, yb).mean()

    model = _train(model, Xn, yj, loss_fn, **kw)
    return model, (mu, sd)


def predict_classes(model, norm, X):
    mu, sd = norm
    logits = jax.vmap(model)((jnp.asarray(X) - mu) / sd)
    return np.asarray(logits.argmax(-1))


class ConvNet(eqx.Module):
    """Compact Equinox CNN for RGB cell images — the conv-net upgrade the module
    docstring promises, for when compact descriptors + MLP trail CNN SOTA."""
    convs: list
    head: eqx.nn.Linear
    pool: eqx.nn.MaxPool2d

    def __init__(self, n_classes, in_ch=3, widths=(32, 64, 128), *, key):
        keys = jax.random.split(key, len(widths) + 1)
        chs = [in_ch] + list(widths)
        self.convs = [eqx.nn.Conv2d(chs[i], chs[i + 1], 3, padding=1, key=keys[i])
                      for i in range(len(widths))]
        self.pool = eqx.nn.MaxPool2d(2, 2)
        self.head = eqx.nn.Linear(widths[-1], n_classes, key=keys[-1])

    def __call__(self, x):                       # x: (C, H, W)
        for conv in self.convs:
            x = self.pool(jax.nn.relu(conv(x)))
        return self.head(jnp.mean(x, axis=(1, 2)))   # global average pool -> head


def augment_images(x, key, color=0.3, geom=True):
    """On-the-fly augmentation for a batch (B,C,H,W). Per-channel gain+bias colour
    jitter simulates the white-balance/stain shift across microscopes (the
    literature fix for WBC domain shift); geom = random flips."""
    k1, k2, k3, k4 = jax.random.split(key, 4)
    if geom:
        x = jnp.where(jax.random.bernoulli(k1, 0.5), x[:, :, :, ::-1], x)
        x = jnp.where(jax.random.bernoulli(k2, 0.5), x[:, :, ::-1, :], x)
    B, C = x.shape[0], x.shape[1]
    gain = 1.0 + color * jax.random.uniform(k3, (B, C, 1, 1), minval=-1.0, maxval=1.0)
    bias = color * jax.random.uniform(k4, (B, C, 1, 1), minval=-1.0, maxval=1.0)
    return x * gain + bias


def train_cnn_classifier(X, y, n_classes, steps=2000, lr=1e-3, batch=64, seed=0,
                         augment=False, color=0.15):
    """X: (N, C, H, W) float images. Returns (model, None).

    augment=True applies colour-jitter + flip augmentation for cross-microscope
    robustness. NOTE: for the Raabin white-balance shift, white-balance-invariant
    preprocessing (eval_external_wbc --color-norm standardize) beats augmentation
    (better Test-B macro-recall, no Test-A cost); strong colour jitter over-
    regularizes. Augmentation is folded into the jitted step (was slow eager)."""
    Xj, yj = jnp.asarray(X), jnp.asarray(y)
    model = ConvNet(n_classes, in_ch=X.shape[1], key=jax.random.PRNGKey(0))
    opt = optax.adamw(lr)
    opt_state = opt.init(eqx.filter(model, eqx.is_array))

    def loss_fn(m, xb, yb):
        logits = jax.vmap(m)(xb)
        return optax.softmax_cross_entropy_with_integer_labels(logits, yb).mean()

    @eqx.filter_jit
    def step(model, opt_state, xb, yb, ak):
        if augment:                                  # python bool -> static under jit
            xb = augment_images(xb, ak, color=color)
        loss, grads = eqx.filter_value_and_grad(loss_fn)(model, xb, yb)
        updates, opt_state = opt.update(grads, opt_state, eqx.filter(model, eqx.is_array))
        return eqx.apply_updates(model, updates), opt_state, loss

    key = jax.random.PRNGKey(seed)
    n = len(Xj)
    for _ in range(steps):
        key, sk, ak = jax.random.split(key, 3)
        idx = jax.random.randint(sk, (min(batch, n),), 0, n)
        model, opt_state, _ = step(model, opt_state, Xj[idx], yj[idx], ak)
    return model, None


def predict_cnn(model, X, batch=256):
    outs = []
    for i in range(0, len(X), batch):
        outs.append(np.asarray(jax.vmap(model)(jnp.asarray(X[i:i + batch])).argmax(-1)))
    return np.concatenate(outs)


def train_regressor(X, y, **kw):
    """y: (N,) or (N, K) marker abundances."""
    Xj = jnp.asarray(X)
    yj = jnp.asarray(y).reshape(len(y), -1)
    mu, sd = Xj.mean(0), Xj.std(0) + 1e-6
    ym, ys = yj.mean(0), yj.std(0) + 1e-6
    Xn, yn = (Xj - mu) / sd, (yj - ym) / ys
    model = MLP(X.shape[1], yj.shape[1], key=jax.random.PRNGKey(1))

    def loss_fn(m, xb, yb):
        return ((jax.vmap(m)(xb) - yb) ** 2).mean()

    model = _train(model, Xn, yn, loss_fn, **kw)
    return model, (mu, sd, ym, ys)
