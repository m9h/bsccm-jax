"""A minimal Lightning logger that logs to trackio (local-first, no login).

Why this exists: VisCy/Lightning trains under the config's `logger:` field, but
Lightning ships no trackio logger, and wiring the wandb logger risks the
interactive wandb-login prompt that hangs headless GPU runs. trackio is a
drop-in, local-first experiment tracker (SQLite + a Gradio dashboard you launch
separately with `trackio show`), so it gives us training/val curves with zero
network/login surface.

Use from a Lightning config:

    trainer:
      logger:
        class_path: bsccm_jax.trackio_logger.TrackioLogger
        init_args:
          project: bsccm-vs

View afterwards on the training host:  `trackio show --project bsccm-vs`
(the run's SQLite db lives under $TRACKIO_DIR, default ~/.cache/huggingface/trackio).
"""

from __future__ import annotations

from lightning.pytorch.loggers.logger import Logger
from lightning.pytorch.utilities.rank_zero import rank_zero_only


def _jsonable(v):
    return isinstance(v, (bool, int, float, str)) or v is None


def _loggable(k, v):
    # trackio reserves config keys starting with "_" (LightningCLI injects
    # "_instantiator"), and only stores JSON scalars.
    return isinstance(k, str) and not k.startswith("_") and _jsonable(v)


class _TrackioExperiment:
    """A tiny TensorBoard/wandb-compatible surface for VisCy's image logging.

    VisCy's `log_image_grid` calls `logger.experiment.add_image(tag, grid, step,
    dataformats="HWC")` (the SummaryWriter API) for any non-wandb logger. Route
    that to trackio, and NEVER raise — sample-image logging must not be able to
    kill a training run (the scalar loss curves are the point).
    """

    def __init__(self, trackio):
        self._t = trackio

    def add_image(self, tag, img, global_step=None, dataformats="HWC"):
        try:
            import numpy as np

            arr = np.asarray(img)
            if dataformats == "CHW" and arr.ndim == 3:
                arr = np.transpose(arr, (1, 2, 0))
            self._t.log({tag: self._t.Image(arr)}, step=global_step)
        except Exception:
            pass

    def log(self, data, step=None):
        try:
            self._t.log(data, step=step)
        except Exception:
            pass


class TrackioLogger(Logger):
    """Log scalar metrics + hyperparameters to a local trackio project.

    Deliberately conservative: trackio is imported lazily and `init` is called
    once (on the first hyperparam/metric event), so importing the module never
    starts a server. Only JSON-scalar hyperparameters are forwarded (the VisCy
    config carries nested model blocks trackio's config store can't hold).
    """

    def __init__(self, project: str = "bsccm-vs", name: str | None = None,
                 save_dir: str = "./trackio_logs"):
        super().__init__()
        self._project = project
        self._run_name = name
        self._save_dir = save_dir
        self._trackio = None
        self._exp = None

    @property
    def name(self) -> str:
        return self._run_name or self._project

    @property
    def version(self) -> str:
        return "0"

    @property
    def save_dir(self) -> str:
        # Lightning derives `trainer.log_dir` (where LightningCLI writes
        # config.yaml, checkpoints, etc.) from a non-TensorBoard logger's
        # save_dir, so this must be a real path, not None.
        return self._save_dir

    @property
    def log_dir(self) -> str:
        return self._save_dir

    @property
    def experiment(self):
        # VisCy accesses logger.experiment.add_image(...) during validation.
        self._ensure()
        if self._exp is None:
            self._exp = _TrackioExperiment(self._trackio)
        return self._exp

    def _ensure(self, config: dict | None = None):
        if self._trackio is None:
            import trackio

            trackio.init(project=self._project, name=self._run_name, config=config or {})
            self._trackio = trackio

    @rank_zero_only
    def log_hyperparams(self, params, *args, **kwargs):
        try:
            d = dict(params) if not isinstance(params, dict) else params
        except (TypeError, ValueError):
            d = vars(params) if hasattr(params, "__dict__") else {}
        self._ensure(config={k: v for k, v in d.items() if _loggable(k, v)})

    @rank_zero_only
    def log_metrics(self, metrics, step=None):
        self._ensure()
        scalars = {k: float(v) for k, v in metrics.items() if v is not None and _jsonable(v)}
        if scalars:
            self._trackio.log(scalars, step=step)

    @rank_zero_only
    def save(self):
        pass

    @rank_zero_only
    def finalize(self, status: str):
        if self._trackio is not None:
            self._trackio.finish()
