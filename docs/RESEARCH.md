# BSCCM computational-microscopy research notes

Reference bibliography and algorithm roadmap for the `bsccm-jax` platform — a
JAX/Kidger + SCICO workbench over the Berkeley Single-Cell Computational
Microscopy dataset. Organized as: (I) foundations we've *implemented*, (II)
latest work from the top groups, (III) additional algorithms to test on BSCCM.

Links marked ✅ were verified during research on 2026-07-07; unlinked entries are
foundational references cited from domain knowledge — confirm before formal use.

---

## I. Foundations — what the current pipeline is built on

**The dataset**
- ✅ Pinkard, Liu, Nyatigo, Fletcher & Waller (2024), *The Berkeley Single Cell
  Computational Microscopy (BSCCM) Dataset*. [arXiv:2402.06191](https://arxiv.org/abs/2402.06191).
  12M+ images of ~400k white blood cells under LED-array coded illumination with
  matched surface-marker fluorescence. Defines our channels (`DPC_Top/Bottom/
  Left/Right`, `DF_*` darkfield ring, 6 fluorescence bands) and benchmark tasks.

**The DPC method we implemented** (`src/bsccm_jax/dpc.py`)
- ✅ Tian & Waller (2015), *Quantitative differential phase contrast imaging in an
  LED array microscope*, Optics Express 23(9):11394.
  [full text](https://opg.optica.org/oe/fulltext.cfm?uri=oe-23-9-11394).
  The 2D weak-object transfer function (WOTF) + half-annulus sources + Tikhonov
  phase deconvolution that `functional_dpc.py` implements and we re-JAX-ified.
  ⚠️ [2023 errata](https://ui.adsabs.harvard.edu/abs/2023OExpr..3111804T/abstract)
  corrects sign/typo errors in the transfer functions — audit `generate_wotf`.
- Mehta & Sheppard (2009), *Quantitative phase-gradient imaging … asymmetric
  illumination-based DPC*, Optics Letters 34(13):1924. The asymmetric-illumination
  DPC foundation.

**The coded-illumination platform**
- Zheng, Horstmeyer & Yang (2013), *Wide-field, high-resolution Fourier
  ptychographic microscopy*, Nature Photonics 7:739. Same LED-array microscope
  class; FPM is the sibling technique on the `coherent` (single-LED) variant.

**The inverse-problem solvers we use**
- ✅ Balke et al. (2022), *Scientific Computational Imaging Code (SCICO)*, JOSS
  7(78):4722. [joss.04722](https://joss.theoj.org/papers/10.21105/joss.04722).
  JAX imaging library; its TV-ADMM gave our best reconstruction (corr 0.9996).
- Boyd et al. (2011), *Distributed Optimization … via ADMM*, Found. Trends ML 3(1).
- Rudin, Osher & Fatemi (1992), *Nonlinear total variation based noise removal*,
  Physica D 60:259. The TV regularizer (our `L21Norm` on the object gradient).

**The Kidger stack (implementation substrate)**
- ✅ Rader, Lyons & Kidger (2023), *Lineax*, [arXiv:2311.17283](https://arxiv.org/abs/2311.17283)
  (NeurIPS AI4Science). Our matrix-free `Normal(CG())` solve.
- ✅ Rader, Lyons & Kidger (2024), *Optimistix: modular optimisation in JAX and
  Equinox*, [arXiv:2402.09983](https://arxiv.org/abs/2402.09983). Our `NonlinearCG`.
- Kidger & Garcia (2021), *Equinox*. The module system for the forward model.

**The headline application BSCCM enables (label-free → fluorescence)**
- Christiansen et al. (2018), *In Silico Labeling*, Cell 173(3):792.
- Ounkomol et al. (2018), *Label-free prediction of 3D fluorescence images …*,
  Nature Methods 15:917.

---

## II. Latest work from the top groups (2024–2026)

**Waller — Computational Imaging Lab, UC Berkeley** ([publications](https://www.laurawaller.com/publications/))
- *Neural space–time model for dynamic multi-shot imaging*, Nature Methods (2024).
  Untrained neural scene+motion model, no pre-training — jointly estimates scene
  and its dynamics. Directly relevant as an object parameterization.
- *Perturbative Fourier ptychographic microscopy for fast quantitative phase
  imaging* (2025). Cuts FPM acquisition/reconstruction cost — applicable to the
  BSCCM coherent variant.
- ✅ *Hybrid-illumination multiplexed Fourier ptychographic microscopy with robust
  aberration correction* (2025). [arXiv:2509.05549](https://arxiv.org/abs/2509.05549).
- *Multi-Modal Deformable Image Registration Using Untrained Neural Networks*,
  ISBI (2025).

**Tian — Computational Imaging Systems Lab, Boston University** ([publications](https://sites.bu.edu/tianlab/publications/))
- *NeuPh: neural phase retrieval* — flexible object representation and
  resolution-enhanced phase from multiplexed FPM.
- *Spatially-varying FourierNet*, Optica (2024). Learned reconstruction with
  space-varying aberrations.
- ✅ *Towards generalizable deep ptychography neural networks* (2025).
  [arXiv:2509.25104](https://arxiv.org/abs/2509.25104). Generalization across
  samples — the key weakness of learned reconstructors.
- ✅ *Refractive index tomography with a physics-based optical neural network*.
  [arXiv:2306.06558](https://arxiv.org/abs/2306.06558).
- *Reflection-mode multi-slice Fourier Ptychographic Tomography*, IEEE Trans.
  Computational Imaging (2026).

**Ozcan — UCLA** (virtual staining / label-free)
- *Virtual Gram staining of label-free bacteria using dark-field microscopy and
  deep learning*, Science Advances (2025). Darkfield → virtual stain — note BSCCM
  ships a full `DF_*` darkfield ring, making this transferable.
  ([summary](https://cnsi.ucla.edu/january-8-2025-ai-powered-staining-in-microbiology-virtual-gram-staining-of-label-free-bacteria/))
- ✅ *Label-free evaluation of lung and heart transplant biopsies using tissue
  autofluorescence-based virtual staining* (2024). [arXiv:2409.05255](https://arxiv.org/abs/2409.05255).

**Generative / diffusion priors for imaging inverse problems**
- ✅ *Poisson–Gaussian Holographic Phase Retrieval with Score-based Image Prior*
  (2023). [arXiv:2305.07712](https://arxiv.org/abs/2305.07712). Score/diffusion
  prior for phase retrieval under realistic noise — close to the DPC setting.
- ✅ *Unleashing the Denoising Capability of Diffusion Prior for Solving Inverse
  Problems* (2024). [arXiv:2406.06959](https://arxiv.org/abs/2406.06959).
- Chung et al., *Diffusion Posterior Sampling (DPS) for general noisy inverse
  problems*, ICLR (2023) — the reference plug-in-prior method.

**Untrained networks / implicit neural representations for QPI**
- ✅ Bostan, Heckel, Chen, Kellman & Waller, *Deep Phase Decoder: self-calibrating
  phase microscopy with an untrained deep neural network*. [arXiv:2001.09803](https://arxiv.org/abs/2001.09803).
  Waller-lab untrained-NN phase + aberration recovery — a natural BSCCM baseline.
- ✅ *Phase imaging with an untrained neural network*, Light: Science &
  Applications (2020). [nature.com/articles/s41377-020-0302-3](https://www.nature.com/articles/s41377-020-0302-3).
- ✅ *Untrained, physics-informed neural networks for structured illumination
  microscopy* (2022). [arXiv:2207.07705](https://arxiv.org/abs/2207.07705).
- ✅ *Deep empirical neural network for optical phase retrieval over a scattering
  medium*, Nature Communications (2025).
  [s41467-025-56522-5](https://www.nature.com/articles/s41467-025-56522-5).
- *FPM image-stack reconstruction using implicit neural representations*, Optica
  (2023); *neural-field-assisted transport-of-intensity QPI*, Photonics Research (2024).
- ✅ Survey + code hub: *Deep learning for phase recovery*, Light: Sci. Appl.
  (2024); Wang et al. resource list [github.com/kqwang/phase-recovery](https://github.com/kqwang/phase-recovery).

---

## III. Additional algorithms to test with BSCCM

Each is tied to what BSCCM provides (redundant multi-contrast measurements +
matched fluorescence/surface markers per cell) and to our JAX/Kidger/SCICO stack.

### A. Better classical / physics-based phase reconstruction
1. **Multi-axis / multiplexed DPC** — go beyond 2-axis (Top/Bottom, Left/Right):
   fold the full `DF_*` darkfield ring into the WOTF for wider frequency coverage
   and more stable low-frequency phase (Tian & Waller's multi-angle result). Pure
   extension of our `annular_sources` + `generate_wotf`.
2. **Richer regularizers in SCICO** — we tested isotropic TV; add anisotropic TV,
   Total Generalized Variation (TGV), Hessian-Schatten, and BM3D/`ppp` priors.
   One-line swaps of the `functional` in our ADMM.
3. **Plug-and-Play priors (PnP-ADMM / PnP-PGM)** — replace the TV prox with a
   learned denoiser (DnCNN, or a JAX diffusion denoiser). SCICO supports this
   directly; bridges classical and learned without paired supervision.
4. **Fourier ptychography on the `coherent` variant** — EPRY-FPM with embedded
   pupil recovery; compare a hand-rolled Equinox/Optimistix implementation
   against the DPC phase on the same cells.

### B. Untrained / self-supervised (no labels — exploits BSCCM's redundancy)
5. **Deep Image Prior / untrained-NN phase** — port Waller's Deep Phase Decoder:
   parameterize the object by an untrained conv net, fit weights to the DPC
   measurements via our differentiable forward model. Equinox net + Optimistix.
6. **Implicit neural representation (INR) object** — swap the pixel-grid
   `(u, p)` for a SIREN/coordinate-MLP; the WOTF forward model stays identical,
   the solve becomes gradient descent over network weights. Gives resolution
   super-sampling and a smoothness prior "for free."
7. **Self-supervised denoising** — Noise2Noise / Noise2Void / equivariant imaging
   across the redundant illumination channels, as a preprocessing or joint prior.

### C. Learned / generative inverse problems
8. **Score/diffusion prior for phase (DPS)** — train a score model on BSCCM phase
   maps, use as a plug-in prior with our differentiable WOTF likelihood. Strong
   fit given the score-based holographic phase-retrieval precedent.
9. **Unrolled networks (learned ADMM/ISTA, "deep unrolling")** — unroll a fixed
   number of our ADMM iterations into a trainable Equinox module; supervise
   against the dataset's precomputed `DPC`/high-NA phase. Physics + data.
10. **Deep equilibrium / fixed-point reconstructors** — Optimistix fixed-point
    solvers make this natural in-stack.

### D. The BSCCM benchmark tasks (label-free → biology)
11. **In-silico labeling / virtual staining** — image-to-image (U-Net, pix2pix,
    or diffusion-based à la Ozcan) from label-free contrast → the 6 fluorescence
    channels. The dataset's headline task.
12. **Surface-marker / cell-type prediction** — regress surface-protein abundance
    or classify white-blood-cell type from label-free channels; BSCCM ships the
    ground-truth labels (`get_surface_marker_data`, `get_cell_type_classification_data`).
13. **Self-supervised representation learning** — contrastive / masked-autoencoder
    pretraining on label-free cells, evaluated on the classification head. Tests
    how much biology is recoverable without stains.

### E. Uncertainty-aware inversion (ties to the lab's JAX-SBI work)
14. **Simulation-based inference over the DPC forward model** — our forward model
    is fully differentiable JAX, so it drops straight into an amortized SBI /
    posterior-estimation pipeline (cf. the group's `sbi4dwi`/BL-1 work): recover a
    *posterior* over phase given noisy measurements, with calibrated uncertainty,
    rather than a point estimate. Novel and directly enabled by what we built.

---

## Immediate next step
Run the real-cell DPC reconstruction on the tiny subset (already on Legion),
compare our JAX phase against the dataset's precomputed `DPC` channel, then pick
one item from §III.B (INR object) or §III.D (in-silico labeling) as the first
extension once the full dataset lands on TrueNAS.
