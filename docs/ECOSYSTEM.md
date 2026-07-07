# The differentiable-microscopy ecosystem — architecture, lineage, and roadmap

Synthesis of the research and design decisions behind `bsccm-jax`. Where
`RESEARCH.md` is the bibliography + algorithm menu, this doc is the **big
picture**: how the pieces (papers, libraries, our code) fit into one JAX-native,
Waller-lab-aligned computational-microscopy pipeline, plus the decisions we made
and why.

Verified links are dated where checked. Treat citations as pointers to confirm.

---

## 1. The one-paragraph thesis

A modern computational microscope is a **differentiable information pipeline**:
you *design* what to measure (maximize information about the object), *simulate*
the optics differentiably, *reconstruct* the object by inverting the forward
model, and optionally *translate* label-free contrast into fluorescence. Every
stage can be autodiff'd, and — crucially for us — **the whole stack has a
JAX-native realization from the Waller/Janelia ecosystem**, which is exactly the
stack `bsccm-jax` is built in (JAX + Kidger + SCICO). Our job is to assemble
these into one platform over the BSCCM single-cell dataset.

## 2. The pipeline, stage by stage (all JAX)

```
   DESIGN            SIMULATE            RECONSTRUCT              TRANSLATE
  ┌────────┐       ┌──────────┐       ┌───────────────┐       ┌──────────┐
  │ IDEAL  │  ───► │ Chromatix│  ───► │  bsccm-jax    │  ───► │ Cytoland │
  │ (info) │       │ (optics) │       │ (inverse)     │       │ (VS)     │
  └────────┘       └──────────┘       └───────────────┘       └──────────┘
  what to          differentiable      DPC/FPM phase           label-free →
  measure          wave-optics fwd     retrieval               fluorescence
```

- **DESIGN — IDEAL** (Pinkard, Kabuli, … Waller; NeurIPS 2025; [arXiv:2405.20559](https://arxiv.org/abs/2405.20559);
  code [Waller-Lab/EncodingInformation](https://github.com/Waller-Lab/EncodingInformation),
  [website](https://waller-lab.github.io/EncodingInformationWebsite/),
  [BAIR blog](https://bair.berkeley.edu/blog/2026/01/10/information-driven-imaging/)).
  "Information-Driven Encoder Analysis Learning": quantify **mutual information**
  between noisy measurements and the object, then gradient-optimize the imaging
  system for it. Universal metric; decoder-free (cheaper than end-to-end).
  *For us:* rank/select which BSCCM channels (Brightfield, DF_* ring, DPC_*)
  carry the most information about phase or a given fluorescence marker.
- **SIMULATE — Chromatix** (Deb, Both, … Waller … Turaga; Nature Methods, Jun 2026;
  `s41592-026-03121-x`). Open-source, GPU-accelerated, **differentiable
  wave-optics library in JAX** (lenses, SLMs, propagation, scattering,
  polarization); 2–22× speedups; demonstrated on snapshot microscopy, holography,
  **phase retrieval**. *For us:* a rigorous forward model — a **JAX-vs-JAX test
  oracle** for our hand-rolled DPC WOTF, the forward engine for FPM (coherent
  variant), and the simulator in the design loop with IDEAL.
- **RECONSTRUCT — bsccm-jax** (this repo). WOTF forward as an Equinox module;
  four solvers (analytic Tikhonov, Lineax matrix-free CG, Optimistix NonlinearCG,
  SCICO TV-ADMM); real-data 2-axis DPC; a per-cell neural field (`neural_field.py`)
  and an amortized self-supervised reconstructor (`amortized.py`).
- **TRANSLATE — Cytoland / VisCy** (Mehta lab, CZ Biohub; Nat. Mach. Intel. 2025;
  [mehta-lab/VisCy](https://github.com/mehta-lab/VisCy)). UNeXt2 virtual staining,
  label-free → fluorescence. We fine-tune it on BSCCM Phase → 6 fluor bands.

## 3. The neural-phase-retrieval lineage (and where our code sits)

One Waller-lab thread runs through several papers you'll keep meeting:

| Paper | Year | Representation | Generalizes? | Our analog |
|---|---|---|---|---|
| **Deep Phase Decoder** (Bostan, Heckel, Chen, Kellman, Waller; Optica 7(6):559) | 2020 | untrained neural field | no (per-sample) | — |
| **NSTM** (Cao & Waller; Nat. Methods) | 2024 | space-time neural field + motion | no (per-acquisition) | `neural_field.py` |
| **PtychoPINN** (Argonne APS) | 2024 | CNN encoder→decoder, physics loss | yes (amortized) | `amortized.py` |
| **NeuPh / LCNF** (Wang…Tian; Adv. Photonics Nexus 3(5):056005) | 2024 | CNN encoder + **conditional** neural field | yes + continuous/super-res | **the target** |

Roots: the untrained-network idea is **Deep Image Prior** (Ulyanov 2018) and
Heckel's **Deep Decoder** (2019, Heckel is a Deep-Phase-Decoder coauthor). Deep
Phase Decoder is the Waller lab's *first* paper on the untrained technique.
**NeuPh is the synthesis of our two prototypes** — amortized encoder (generalizes)
+ conditional neural-field decoder (continuous phase) — and is the architecture
to grow `amortized.py` toward, on the FPM/coherent variant.

## 4. Test-oracle & cross-validation strategy

We validate our from-scratch JAX solvers against trusted references:

- **Chromatix (JAX)** — cleanest oracle: simulate the same LED-array system,
  cross-check our WOTF forward and reconstruction. JAX-vs-JAX.
- **DeepInverse `deepinv` (PyTorch)** — mature inverse-problems library; wrap our
  DPC WOTF as a `LinearPhysics`, compare our TV-ADMM/PGD to `deepinv.optim`. Also
  the source of **self-supervised losses** (Equivariant Imaging, measurement
  consistency, SURE) to strengthen `amortized.py`, and `deepinv.sampling` (DPS/
  DiffPIR) if we ever want a diffusion prior — no custom build needed.
- **PtychoPINN (PyTorch)** — reference for amortized self-supervised ptychography;
  cloned at `~/Workspace/PtychoPINN-torch-pub` with a working CPU-torch venv.

## 5. Decisions & assessments (so we don't relitigate)

- **Diffusion priors: SKIP for DPC.** DPC is near-linear and well-conditioned
  (multi-axis diversity); classical + conditional-neural-field win on
  quality-per-compute. Diffusion earns its keep only where it's severely
  ill-posed/nonlinear — the **FPM/coherent variant** — or on the **generative
  virtual-staining output** (Ozcan's diffusion VS). Ref that prompted this:
  Poisson-Gaussian holographic phase retrieval w/ score prior ([arXiv:2305.07712](https://arxiv.org/abs/2305.07712)),
  and ProjDiff (NeurIPS 2024, [weigerzan/ProjDiff](https://github.com/weigerzan/ProjDiff)).
- **NSTM: don't full-port.** Its differentiator is *motion* estimation for
  dynamic multi-shot; BSCCM cells are static, so the lean Equinox neural field
  (the useful subset) is kept, the motion machinery skipped. On static cells it
  ≈ Tikhonov (corr ~0.55), no gain — so the target is amortized/NeuPh, not NSTM.
- **Amortized > per-cell at scale.** Our four solvers and the neural field all
  optimize per cell; BSCCM has 400k cells. The amortized self-supervised model
  (train once, single-pass inference, generalizes) is the scale story — the
  PtychoPINN paradigm, growing toward NeuPh's conditional neural field.
- **No pretrained BSCCM model exists.** The dataset ships tasks + labels
  (`get_cell_type_classification_data`, `get_surface_marker_data`) but no weights.
  Closest released code: multi-task phenotyping ([saqibnaziir/Single-Cell-Phenotyping](https://github.com/saqibnaziir/Single-Cell-Phenotyping),
  DPC→WBC-class 91.3% + CD16 regression 0.72) — code, not checkpoints. So we
  train/fine-tune ourselves.

## 6. In-silico labeling — current status (WORKING)

Cytoland/VisCy UNeXt2 fine-tune, BSCCM `Phase` → 6 `Fluor_*` bands, training on
the **DGX Spark (GB10 Blackwell)** inside NGC `nvcr.io/nvidia/pytorch:26.06-py3`.
See the hard-won setup gotchas in memory (`project_bsccm_jax`). Pipeline:
`scripts/bsccm_to_omezarr.py` → paired HCS OME-Zarr → `generate_normalization_metadata`
→ `configs/bsccm_vs_finetune.yml` → `viscy fit` (detached, stdin closed, wandb off).
Next: **trackio** (local, login-free, wandb-compatible) via Lightning's WandbLogger
for loss curves + predicted-vs-true fluorescence image grids; then eval on val cells.

## 7. Educational direction — a two-part computational-imaging curriculum

Two short Waller-lab papers form a clean curriculum, and `bsccm-jax` is the JAX
substrate to teach both on real single-cell data (browser/zero-hardware):
- **Part 1 — "what to measure": IDEAL** (below). Imaging as an information channel.
- **Part 2 — "how to reconstruct": "How to do Physics-based Learning"**
  (Kellman, Lustig & Waller 2020, [arXiv:2005.13531](https://arxiv.org/abs/2005.13531)).
  A 3-page tutorial: use autodiff through the forward model *twice* (to build the
  reconstruction net, then to train it) — implement only the physics. This is
  literally what `dpc.py` + `neural_field.py` + `amortized.py` demonstrate in JAX.

IDEAL reframes imaging as an **information channel** — design = maximize mutual
information, not visual prettiness — which is the single idea that explains *why
computational imaging works*. It's already packaged for teaching (open code,
notebooks, BAIR blog). Proposed module, grounded in BSCCM real data:
students pick measurement subsets → estimate mutual information about the object
→ **see** reconstruction/virtual-staining quality track the information, in
napari. Lab exercise: "find the 4 most informative LED patterns for predicting
CD16." **Browser-deliverable, zero-hardware** (data + simulation) — aligns with
the remote student-access work (browser-only access for students in Africa): a
rigorous imaging course with no microscope required.

## 8. Hardware direction (live rig)

See memory `project_bsccm_jax` for detail. Two modalities, kept separate:
(A) **LED-array coded illumination (DPC/FPM)** — what `dpc.py` reconstructs; the
natural OpenFlexure mod is a 32×32 APA102 matrix + `illuminate` firmware
(flat array suffices for DPC; quasi-dome for FPM). (B) **Spectral DiffuserScope**
snapshot hyperspectral fluorescence — SCICO/TV turf, but a separate optical
module, not an OpenFlexure retrofit. Frontier tying it together: **differentiable
coded-illumination design** — autodiff through the microscope (Chromatix) to
optimize LED patterns (IDEAL objective), then reconstruct (bsccm-jax).

---

## Component inventory

| Piece | Where | Status |
|---|---|---|
| DPC reconstruction (4 solvers) | `src/bsccm_jax/dpc.py` | done, tested |
| Per-cell neural field | `src/bsccm_jax/neural_field.py` | done |
| Amortized self-supervised recon | `src/bsccm_jax/amortized.py` | prototype (GPU needed) |
| napari view | `src/bsccm_jax/view.py` | done |
| Dryad OAuth downloader | `scripts/dryad_download.py` | done; 197 GB main fetched |
| BSCCM → OME-Zarr converter | `scripts/bsccm_to_omezarr.py` | done |
| Cytoland fine-tune config | `configs/bsccm_vs_finetune.yml` | training on GB10 |
| Chromatix forward/oracle | (to adopt) | planned |
| IDEAL teaching module | (to build) | planned |
