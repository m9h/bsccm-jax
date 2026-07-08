"""Modal app — cloud training + end-to-end repo validation for bsccm-jax.

Runs the pipeline on Modal from a FRESH CLONE of the public repo, which both
(a) trains on the full BSCCM dataset without the local-infra pain, and
(b) is the cleanest possible test that the repo installs and runs from scratch.

Setup (once):
    modal setup                      # auth (already done on the DGX Spark)
    modal secret create dryad-creds \
        DRYAD_CLIENT_ID=xxxx DRYAD_CLIENT_SECRET=yyyy

Run:
    modal run modal_app.py::test_repo            # clone + install + pytest + self-tests
    modal run modal_app.py::download --full      # Dryad -> persistent Modal Volume
    modal run modal_app.py::train_phenotype      # Tier 1 classification+regression on full data
    modal run modal_app.py::reconstructions      # FPM + NeuPh validation on a GPU
"""

import subprocess

import modal

REPO = "https://github.com/m9h/bsccm-jax"

# The image bakes a fresh clone + `uv sync` — so building the image IS the repo
# install test. `uv run` in each function uses the resulting project venv.
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "build-essential")
    .pip_install("uv")
    .run_commands(
        f"git clone {REPO} /root/bsccm-jax",
        "cd /root/bsccm-jax && uv sync",
    )
    .workdir("/root/bsccm-jax")
)

app = modal.App("bsccm-jax")
vol = modal.Volume.from_name("bsccm-data", create_if_missing=True)


def _run(cmd):
    subprocess.run(cmd, shell=True, cwd="/root/bsccm-jax", check=True)


@app.function(image=image, timeout=1800)
def test_repo():
    """Validate the repo end-to-end: the test suite + the DPC self-test."""
    _run("uv run pytest tests/ -q")
    _run("uv run python -m bsccm_jax.selftest")
    print("REPO OK: tests pass and the DPC reconstructors self-test cleanly.")


@app.function(image=image, timeout=1800)
def reconstructions():
    """Exercise the FPM + NeuPh reconstructors (synthetic) — validates Tier 2/3."""
    _run(
        "uv run python -c \""
        "import numpy as np; from bsccm_jax import dpc, fpm, neuph; "
        "hr=fpm.hr_phantom((256,256)); pup=fpm.circ_pupil((64,64),9); sh=fpm.led_grid_shifts(9,6); "
        "im=fpm.fpm_forward(hr,sh,pup,(64,64)); rec=fpm.reconstruct_fpm(im,sh,pup,(256,256),steps=300); "
        "print('FPM ok'); "
        "Hp=dpc.dpc_2axis_transfer((64,64),wavelength_um=0.532,pixel_size_um=0.2,na=0.5); "
        "d=[ (np.asarray(dpc.dpc_apply_phase(Hp,dpc.phantom((64,64),seed=s)[1])), np.asarray(dpc.phantom((64,64),seed=s)[1])) for s in range(40)]; "
        "import numpy as np; M=np.stack([x[0] for x in d]); P=np.stack([x[1] for x in d]); "
        "m,c=neuph.train_lcnf(M,P,epochs=40); print('NeuPh held-out corr', round(c,3))\""
    )


@app.function(image=image, volumes={"/data": vol},
              secrets=[modal.Secret.from_name("dryad-creds")], timeout=6 * 3600)
def download(full: bool = False, coherent: bool = False, mnist: bool = False):
    """Download BSCCM from Dryad into the persistent Modal Volume (fast cloud net)."""
    flags = " ".join(f for f, on in
                     [("--full", full), ("--coherent", coherent), ("--mnist", mnist)] if on)
    _run(f"uv run python scripts/dryad_download.py --location /data/ {flags}")
    vol.commit()
    print("download complete -> /data (committed to the bsccm-data volume)")


@app.function(image=image, cpu=8.0, memory=32768, volumes={"/data": vol}, timeout=3 * 3600)
def train_phenotype(data_path: str = "/data/BSCCM"):
    """Tier 1: WBC classification + surface-marker regression on the full dataset."""
    _run(f"uv run python scripts/train_phenotype.py --data {data_path} --n 6000")


@app.function(image=image, cpu=16.0, memory=65536, volumes={"/data": vol}, timeout=4 * 3600)
def stage_staining_data(data_path: str = "/data/BSCCM", n: int = 20000,
                        out: str = "/data/bsccm_vs.zarr"):
    """Convert full BSCCM -> paired HCS OME-Zarr (Phase + fluor) on the Volume."""
    _run(f"uv run python scripts/bsccm_to_omezarr.py --data {data_path} --out {out} --n {n}")
    vol.commit()


# torch + VisCy/Cytoland image for the virtual-staining fine-tune (standard CUDA
# GPUs on Modal -> plain torch wheels, no Blackwell container needed).
staining_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "build-essential")
    .pip_install("uv")
    .run_commands(
        "uv pip install --system torch --index-url https://download.pytorch.org/whl/cu124",
        "git clone https://github.com/mehta-lab/VisCy /root/VisCy",
        "cd /root/VisCy && uv pip install --system "
        "-e packages/viscy-utils -e packages/viscy-transforms "
        "-e packages/viscy-data -e packages/viscy-models -e applications/cytoland",
        "git clone https://github.com/m9h/bsccm-jax /root/bsccm-jax",
    )
    .workdir("/root/bsccm-jax")
)


@app.function(image=staining_image, gpu="a100", volumes={"/data": vol}, timeout=8 * 3600)
def train_staining(zarr: str = "/data/bsccm_vs.zarr", max_epochs: int = 100):
    """In-silico labeling: fine-tune Cytoland/VSUNet Phase->6-fluor on the full data."""
    import subprocess
    # 1) normalization metadata (NormalizeSampled needs it)
    subprocess.run(
        "python -c \"from viscy_utils.meta_utils import generate_normalization_metadata as g; "
        f"g('{zarr}', num_workers=8)\"", shell=True, check=True)
    # 2) fine-tune (config data_path/epochs overridden; wandb off, logger off)
    subprocess.run(
        f"WANDB_MODE=disabled viscy fit -c configs/bsccm_vs_finetune.yml "
        f"--data.init_args.data_path={zarr} --trainer.max_epochs={max_epochs} "
        f"--trainer.default_root_dir=/data/vs_out --trainer.enable_progress_bar=false",
        shell=True, cwd="/root/bsccm-jax", check=True)
    vol.commit()
    print("virtual-staining model trained -> /data/vs_out (committed)")


@app.function(image=image, cpu=8.0, memory=32768, volumes={"/data": vol}, timeout=3 * 3600)
def benchmark(data_path: str = "/data/BSCCM"):
    """Run the full-dataset benchmark and print the consolidated results table."""
    _run(f"uv run python scripts/run_benchmark.py --data {data_path} --n 6000 --out /data/benchmark_results.json")
    vol.commit()


@app.local_entrypoint()
def main():
    # default: validate the repo, then the Tier 2/3 reconstructors
    test_repo.remote()
    reconstructions.remote()
