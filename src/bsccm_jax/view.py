"""napari view of the DPC computational-microscopy pipeline.

Assembles aligned layers — raw coded (DPC) measurements, the reconstructed
quantitative phase and absorption, and (when available) the ground-truth object
or matched fluorescence — so the whole "coded measurements in, quantitative
contrast out" story reads in one window.

    uv run python -m bsccm_jax.view                 # phantom, opens napari
    uv run python -m bsccm_jax.view --screenshot out.png   # headless render

Real BSCCM data slots in unchanged once the dataset lands (see load_bsccm()).
"""

import argparse

import jax.numpy as jnp
import numpy as np

from bsccm_jax import dpc


def build_layers(method="scico"):
    """Return (layers, viewer_title). Uses the synthetic phantom as the object."""
    shape = (128, 128)
    u_true, p_true = dpc.phantom(shape)
    fwd = dpc.DPCForward.build(shape)
    images = fwd(u_true, p_true)

    recon = {
        "scico": dpc.reconstruct_scico,
        "tikhonov": dpc.reconstruct_tikhonov,
        "lineax": dpc.reconstruct_lineax,
        "optimistix": dpc.reconstruct_optimistix,
    }[method]
    _, p_hat = recon(images, fwd)

    layers = [
        (np.asarray(images), {"name": "coded DPC measurements", "colormap": "gray"}, "image"),
        (np.asarray(p_hat), {"name": f"reconstructed phase [{method}]", "colormap": "viridis"}, "image"),
        (np.asarray(p_true), {"name": "ground-truth phase", "colormap": "viridis",
                              "visible": False}, "image"),
    ]
    return layers, f"BSCCM DPC — {method} reconstruction"


def load_bsccm(data_root, index=None, channel="DPC"):
    """Load real BSCCM images once the dataset is on disk (e.g. from TrueNAS)."""
    from bsccm import BSCCM

    data = BSCCM(data_root, cache_index=True)
    idx = int(index) if index is not None else int(data.get_indices()[0])
    dpc_img = np.asarray(data.read_image(idx, channel, copy=True))
    fluor = None
    try:
        fchan = data.fluor_channel_names[0]
        fluor = np.asarray(data.read_image(idx, fchan, copy=True))
    except Exception:
        pass
    return dpc_img, fluor, idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", default="scico",
                    choices=["scico", "tikhonov", "lineax", "optimistix"])
    ap.add_argument("--screenshot", metavar="PNG", default=None,
                    help="render offscreen to PNG instead of opening a window")
    args = ap.parse_args()

    if args.screenshot:
        import os
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

    import napari

    layers, title = build_layers(args.method)
    viewer = napari.Viewer(title=title, show=not args.screenshot)
    for data, kw, _kind in layers:
        viewer.add_image(data, **kw)
    viewer.grid.enabled = True
    viewer.reset_view()

    if args.screenshot:
        viewer.screenshot(args.screenshot, canvas_only=True, flash=False, size=(900, 400))
        print(f"wrote {args.screenshot}")
        viewer.close()
    else:
        napari.run()


if __name__ == "__main__":
    main()
