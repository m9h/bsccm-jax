# bsccm-jax

A JAX / Kidger-stack computational-microscopy workbench over the **Berkeley
Single-Cell Computational Microscopy (BSCCM)** dataset. Reconstructs quantitative
phase from differential-phase-contrast (DPC) coded-illumination images, and
visualizes the pipeline in napari.

This is a standalone project — **not** a fork. It uses BSCCM as a data source and
draws methodologically on Tian & Waller DPC and NSTM (see [NOTICE](NOTICE)).

## What's here

- **`src/bsccm_jax/dpc.py`** — the DPC weak-object transfer function (WOTF) forward
  model as an Equinox module, plus four reconstructors: analytic Tikhonov, Lineax
  matrix-free CG, Optimistix NonlinearCG, and SCICO TV-ADMM. Re-JAX-ified from
  BSCCM's `functional_dpc.py`. Also the real-data 2-axis normalized DPC solve.
- **`src/bsccm_jax/neural_field.py`** — NSTM's neural-field object representation
  reimplemented in Equinox + Optax (coordinate MLP + annealed positional
  encoding), composed with the DPC forward model above.
- **`src/bsccm_jax/view.py`** — napari layers (measurements, phase, ground truth).
- **`scripts/dryad_download.py`** — resumable BSCCM download via Dryad OAuth2.
- **`scripts/real_cell_demo.py`** — real-cell reconstruction + validation figure.
- **`tests/test_dpc.py`** — red-green TDD spec (all reconstructors vs a phantom).
- **`docs/RESEARCH.md`** — bibliography + roadmap of algorithms to test on BSCCM.

## Stack

Runs on the latest JAX (0.10.2). Kidger: equinox, optimistix, lineax, diffrax,
jaxtyping. Imaging solver: SCICO (jax pin overridden — see `pyproject.toml`).
Data + viz: bsccm (editable), napari, zarr, dask.

## Quickstart

```bash
uv sync                                   # install the environment
uv run pytest tests/ -q                   # verify all reconstructors (5 tests)
# get data (needs Dryad OAuth creds in ~/.config/dryad/credentials):
uv run python scripts/dryad_download.py --location data/          # tiny subset
uv run python scripts/real_cell_demo.py --n 4 --out real_cell_dpc.png
uv run python -m bsccm_jax.view --method scico                    # napari
```

## License

BSD-3-Clause (see [LICENSE](LICENSE) and [NOTICE](NOTICE) for upstream credits).
