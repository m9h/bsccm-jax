"""Differential phase contrast (DPC) reconstruction, JAX/Kidger-stack edition.

This is a faithful re-JAX-ification of ``bsccm/phase/functional_dpc.py`` (whose
``import numpy as np; import numpy as onp`` fingerprint shows it began life as JAX
code). The weak-object transfer function (WOTF) forward model is *linear* in the
object's absorption ``u`` and phase ``p``:

    I_k = Re( iF( Hu_k * F(u) + Hp_k * F(p) ) )

so it maps cleanly onto the Kidger stack in three complementary ways:

  * ``DPCForward``            — the optics as an ``equinox.Module`` (fixed Hu/Hp,
                               callable on a candidate object).
  * ``reconstruct_tikhonov`` — the analytic per-frequency 2x2 closed form.
  * ``reconstruct_lineax``   — matrix-free least squares: hand Lineax the forward
                               operator and let ``NormalCG`` form the adjoint by
                               autodiff. No hand-derived gradient.
  * ``reconstruct_optimistix`` — regularised nonlinear solve of the very same
                               ``dpc_loss`` residual, gradients via ``jax.grad``.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
import optimistix as optx
from jaxtyping import Array, Complex, Float

jax.config.update("jax_enable_x64", True)  # phase retrieval wants the precision

# --- Fourier operators (match the package's shifted convention) --------------
F = lambda x: jnp.fft.ifftshift(jnp.fft.fft2(jnp.fft.fftshift(x)))
iF = lambda x: jnp.fft.ifftshift(jnp.fft.ifft2(jnp.fft.fftshift(x)))


# --- Optics: spatial-frequency grid, pupil, half-annulus sources, WOTF -------
def spatial_freq_grid(shape, pixel_size_um):
    fov = jnp.asarray(shape) * pixel_size_um
    dfx, dfy = 1.0 / fov
    fx = dfx * (jnp.arange(shape[0]) - jnp.round((shape[0] + 1) / 2))
    fy = dfy * (jnp.arange(shape[1]) - jnp.round((shape[1] + 1) / 2))
    fy_g, fx_g = jnp.meshgrid(fy, fx)
    return fx_g, fy_g


def pupil_support(wavelength_um, na, pixel_size_um, shape, inner_na=None):
    fx, fy = spatial_freq_grid(shape, pixel_size_um)
    fmax = na / wavelength_um
    r = jnp.sqrt((fx / fmax) ** 2 + (fy / fmax) ** 2)
    if inner_na is None:
        return r < 1
    return jnp.logical_and(r < 1, r > inner_na / wavelength_um)


def annular_sources(wavelength_um, pixel_size_um, shape, na, rotations_deg, na_inner=0.0):
    """Half-annulus (asymmetric) illumination patterns — the DPC source set."""
    fx, fy = spatial_freq_grid(shape, pixel_size_um)
    ring = pupil_support(wavelength_um, na, pixel_size_um, shape, inner_na=na_inner or None)
    src = []
    for rot in rotations_deg:
        half = (jnp.cos(jnp.deg2rad(rot)) * fx + 1e-15) >= (jnp.sin(jnp.deg2rad(rot)) * fy)
        src.append((ring * half).astype(jnp.float64))
    return jnp.stack(src)


def generate_pupil(wavelength_um, pixel_size_um, na, shape):
    return pupil_support(wavelength_um, na, pixel_size_um, shape).astype(jnp.complex128)


def generate_wotf(sources: Float[Array, "k h w"], pupil: Complex[Array, "h w"]):
    """Weak-object transfer functions (amplitude Hu, phase Hp) per source."""
    def one(src):
        I0 = (src * pupil * pupil.conj()).sum()
        term1 = F(pupil).conj() * F(src * pupil)
        term2 = F(src * pupil.conj()).conj() * F(pupil.conj())
        hu = iF(term1 + term2) / I0
        hp = 1j * iF(term1 - term2) / I0
        return hu, hp

    Hu, Hp = jax.vmap(one)(sources.astype(jnp.complex128))
    return Hu, Hp


# --- The forward model as an Equinox module ----------------------------------
class DPCForward(eqx.Module):
    """Fixed optics; call on a candidate object (u, p) -> stack of DPC images."""

    Hu: Complex[Array, "k h w"]
    Hp: Complex[Array, "k h w"]

    def __call__(self, u: Float[Array, "h w"], p: Float[Array, "h w"]) -> Float[Array, "k h w"]:
        fu, fp = F(u), F(p)
        return jax.vmap(lambda hu, hp: iF(hu * fu + hp * fp).real)(self.Hu, self.Hp)

    @classmethod
    def build(cls, shape, *, wavelength_um=0.532, pixel_size_um=0.2, na=0.4,
              rotations_deg=(0.0, 180.0, 90.0, 270.0)):
        pupil = generate_pupil(wavelength_um, pixel_size_um, na, shape)
        sources = annular_sources(wavelength_um, pixel_size_um, shape, na, rotations_deg)
        Hu, Hp = generate_wotf(sources, pupil)
        return cls(Hu=Hu, Hp=Hp)


# --- Reconstruction method 1: analytic Tikhonov (closed form) ----------------
def reconstruct_tikhonov(dpc_images, fwd: DPCForward, reg_u=1e-1, reg_p=5e-3):
    Hu, Hp = fwd.Hu, fwd.Hp
    AHA = [(Hu.conj() * Hu).sum(0) + reg_u, (Hu.conj() * Hp).sum(0),
           (Hp.conj() * Hu).sum(0), (Hp.conj() * Hp).sum(0) + reg_p]
    det = AHA[0] * AHA[3] - AHA[1] * AHA[2]
    fI = jax.vmap(F)(dpc_images.astype(jnp.complex128))
    AHy = jnp.stack([(Hu.conj() * fI).sum(0), (Hp.conj() * fI).sum(0)])
    u = iF((AHA[3] * AHy[0] - AHA[1] * AHy[1]) / det).real
    p = iF((AHA[0] * AHy[1] - AHA[2] * AHy[0]) / det).real
    return u, p


# --- Reconstruction method 2: Lineax matrix-free least squares ---------------
def reconstruct_lineax(dpc_images, fwd: DPCForward, reg=1e-2, **kw):
    """Solve the Tikhonov normal equations with a matrix-free CG.

    The adjoint of the WOTF operator is never written by hand — Lineax's
    ``Normal(CG())`` obtains it from the forward map via autodiff. (Requires
    jax != 0.7.0/0.7.1, which equinox blacklists and where this path breaks.)
    """
    shape = dpc_images.shape[1:]
    struct = (jax.ShapeDtypeStruct(shape, jnp.float64),
              jax.ShapeDtypeStruct(shape, jnp.float64))

    def forward(up):
        u, p = up
        pred = fwd(u, p)
        # Stack a Tikhonov penalty onto the residual so NormalCG regularises.
        return (pred, jnp.sqrt(reg) * u, jnp.sqrt(reg) * p)

    op = lx.FunctionLinearOperator(forward, struct)
    rhs = (dpc_images, jnp.zeros(shape), jnp.zeros(shape))
    sol = lx.linear_solve(op, rhs, solver=lx.Normal(lx.CG(rtol=1e-6, atol=1e-6, **kw)))
    u, p = sol.value
    return u, p


# --- Reconstruction method 3: Optimistix nonlinear solve of dpc_loss ---------
def dpc_loss(u, p, dpc_images, fwd: DPCForward, reg_u=1e-1, reg_p=5e-3):
    resid = dpc_images - fwd(u, p)
    reg = iF(reg_u * F(u) + reg_p * F(p)).real ** 2
    return jnp.sum(resid ** 2) + jnp.sum(reg)


def reconstruct_optimistix(dpc_images, fwd: DPCForward, reg_u=1e-1, reg_p=5e-3,
                           max_steps=256):
    """Matrix-free nonlinear CG on ``dpc_loss``; gradients via ``jax.grad``.

    NonlinearCG (not BFGS): a 128x128 object is 32k parameters, so a dense
    inverse-Hessian is infeasible — CG keeps it matrix-free.
    """
    shape = dpc_images.shape[1:]

    def objective(params, _):
        u, p = params
        return dpc_loss(u, p, dpc_images, fwd, reg_u, reg_p)

    solver = optx.NonlinearCG(rtol=1e-8, atol=1e-8)
    init = (jnp.zeros(shape), jnp.zeros(shape))
    sol = optx.minimise(objective, solver, init, max_steps=max_steps, throw=False)
    u, p = sol.value
    return u, p


# --- Reconstruction method 4: SCICO TV-regularized ADMM ----------------------
def reconstruct_scico(dpc_images, fwd: DPCForward, reg_tv=5e-3, rho=1.0,
                      max_iter=100):
    """Total-variation phase reconstruction via SCICO's ADMM.

    Purpose-built imaging solver: the WOTF forward map becomes a SCICO
    ``LinearOperator``, the data term an L2 loss, and an anisotropic-TV
    functional the regularizer — giving edge-preserving phase that the plain L2
    Tikhonov / Lineax solves smear. This is the imaging-specific solver role.
    """
    from scico import functional, linop, loss
    from scico.optimize.admm import ADMM, LinearSubproblemSolver

    shape = dpc_images.shape[1:]
    obj_shape = (2, *shape)  # channel 0 = absorption u, channel 1 = phase p
    y = jnp.asarray(dpc_images, jnp.float64)

    def eval_fn(x):
        return fwd(x[0], x[1])  # linear WOTF map; autodiff supplies the adjoint

    A = linop.LinearOperator(input_shape=obj_shape, output_shape=dpc_images.shape,
                             eval_fn=eval_fn, input_dtype=jnp.float64,
                             output_dtype=jnp.float64, jit=True)
    f = loss.SquaredL2Loss(y=y, A=A)
    C = linop.FiniteDifference(input_shape=obj_shape, input_dtype=jnp.float64,
                               axes=(1, 2), circular=True)
    g = reg_tv * functional.L21Norm()  # isotropic TV on the object gradient

    solver = ADMM(f=f, g_list=[g], C_list=[C], rho_list=[rho], x0=A.T @ y,
                  subproblem_solver=LinearSubproblemSolver(cg_kwargs={"maxiter": 50}),
                  maxiter=max_iter)
    x = solver.solve()
    return x[0], x[1]


# --- Shared real-data DPC operator (reused by the neural-field reconstructor) -
def dpc_2axis_transfer(shape, *, wavelength_um=0.532, pixel_size_um=0.2, na=0.5):
    """Differential phase transfer functions Hp (2, H, W) for Top/Bottom & Left/Right."""
    src = annular_sources(wavelength_um, pixel_size_um, shape, na, (90., 270., 180., 0.))
    pupil = generate_pupil(wavelength_um, pixel_size_um, na, shape)
    _, Hp_each = generate_wotf(src, pupil)
    return jnp.stack([Hp_each[0] - Hp_each[1], Hp_each[2] - Hp_each[3]])


def dpc_measurements(imgs):
    """Normalized differential measurements (2, H, W) from raw T/B/L/R images."""
    T, B, L, R = (jnp.asarray(imgs[k], jnp.float64) for k in ("Top", "Bottom", "Left", "Right"))

    def diff(a, b):
        s = a + b
        return jnp.where(s > 0, (a - b) / s, 0.0)

    return jnp.stack([diff(T, B), diff(L, R)])


def dpc_apply_phase(Hp, phase):
    """Forward: phase field -> predicted differential measurements."""
    fp = F(phase.astype(jnp.complex128))
    return jax.vmap(lambda h: iF(h * fp).real)(Hp)


# --- Real-data DPC: 2-axis normalized differential phase reconstruction -------
def reconstruct_dpc_2axis(imgs, *, wavelength_um=0.532, pixel_size_um=0.2,
                          na=0.5, reg=5e-3, tv=False, tv_weight=2e-3, max_iter=100):
    """Phase from real Top/Bottom/Left/Right half-annulus intensity images.

    Standard Tian–Waller DPC: normalize each opposing pair to a differential
    measurement d = (I_a - I_b)/(I_a + I_b), build the differential-source WOTF,
    and deconvolve phase. Absorption is dropped (it cancels in the normalized
    difference), so this is a phase-only solve.

    imgs: dict with keys 'Top','Bottom','Left','Right' (2D real arrays).
    tv=True routes the phase solve through SCICO TV-ADMM instead of Tikhonov.
    Returns the reconstructed phase (H, W).
    """
    T, B, L, R = (jnp.asarray(imgs[k], jnp.float64) for k in ("Top", "Bottom", "Left", "Right"))
    shape = T.shape

    def diff(a, b):
        s = a + b
        return jnp.where(s > 0, (a - b) / s, 0.0)

    meas = jnp.stack([diff(T, B), diff(L, R)])  # (2, H, W)

    # Individual half-annulus sources (each a physical, non-negative intensity so
    # its WOTF normalization 1/I0 is well posed), then difference the *transfer
    # functions* per axis. Differencing the sources directly would give I0≈0.
    # rotations: axis 0 splits along fy (Top/Bottom), axis 1 along fx (Left/Right).
    src = annular_sources(wavelength_um, pixel_size_um, shape, na, (90., 270., 180., 0.))
    pupil = generate_pupil(wavelength_um, pixel_size_um, na, shape)
    _, Hp_each = generate_wotf(src, pupil)  # phase transfer per individual source
    Hp = jnp.stack([Hp_each[0] - Hp_each[1], Hp_each[2] - Hp_each[3]])  # (2, H, W)

    if not tv:
        # Phase-only Tikhonov: p = iF( Σ conj(Hp)·F(d) / (Σ|Hp|² + reg) )
        fmeas = jax.vmap(F)(meas.astype(jnp.complex128))
        num = (Hp.conj() * fmeas).sum(0)
        den = (Hp.conj() * Hp).sum(0) + reg
        return iF(num / den).real

    # TV via SCICO: forward operator phase -> [d_tb, d_lr]
    from scico import functional, linop, loss
    from scico.optimize.admm import ADMM, LinearSubproblemSolver

    def eval_fn(p):
        fp = F(p.astype(jnp.complex128))
        return jax.vmap(lambda h: iF(h * fp).real)(Hp)

    A = linop.LinearOperator(input_shape=shape, output_shape=meas.shape,
                             eval_fn=eval_fn, input_dtype=jnp.float64,
                             output_dtype=jnp.float64, jit=True)
    f = loss.SquaredL2Loss(y=jnp.asarray(meas, jnp.float64), A=A)
    C = linop.FiniteDifference(input_shape=shape, input_dtype=jnp.float64, circular=True)
    g = tv_weight * functional.L21Norm()
    solver = ADMM(f=f, g_list=[g], C_list=[C], rho_list=[1.0], x0=A.T @ meas,
                  subproblem_solver=LinearSubproblemSolver(cg_kwargs={"maxiter": 50}),
                  maxiter=max_iter)
    return solver.solve()


# --- Synthetic phantom so the whole pipeline is provable without the dataset -
def phantom(shape=(128, 128), seed=0):
    """A smooth phase object + faint absorption — a stand-in BSCCM 'cell'."""
    key = jax.random.PRNGKey(seed)
    ky, ku = jax.random.split(key)
    yy, xx = jnp.mgrid[0:shape[0], 0:shape[1]] / jnp.asarray(shape)[:, None, None]

    def blobs(k, n):
        c = jax.random.uniform(k, (n, 2), minval=0.2, maxval=0.8)
        s = jax.random.uniform(k, (n,), minval=0.02, maxval=0.06)
        a = jax.random.uniform(k, (n,), minval=0.5, maxval=1.0)
        img = jnp.zeros(shape)
        for i in range(n):
            r2 = (xx - c[i, 1]) ** 2 + (yy - c[i, 0]) ** 2
            img = img + a[i] * jnp.exp(-r2 / (2 * s[i] ** 2))
        return img

    p = blobs(ky, 5)              # phase: the thing we mainly want back
    u = 0.15 * blobs(ku, 3)       # absorption: faint
    return u, p
