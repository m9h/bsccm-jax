"""FPM demo: recover a high-res complex object from low-res LED-array images.

Forward-simulates the coherent (single-LED) BSCCM-style acquisition, reconstructs
with the differentiable gradient solver, and renders the super-resolution story:
true object vs a single objective-limited image vs the FPM reconstruction.
"""

import jax
import jax.numpy as jnp
import numpy as np

from bsccm_jax import fpm


def corr(a, b):
    a = np.asarray(a).ravel() - np.mean(a); b = np.asarray(b).ravel() - np.mean(b)
    return float((a * b).sum() / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def main():
    HR, LR = (256, 256), (64, 64)
    hr = fpm.hr_phantom(HR)
    pupil = fpm.circ_pupil(LR, radius=9)
    shifts = fpm.led_grid_shifts(9, 6)
    imgs = fpm.fpm_forward(hr, shifts, pupil, LR)
    rec = fpm.reconstruct_fpm(imgs, shifts, pupil, HR, steps=400, lr=3e-2)

    single = np.asarray(jax.image.resize(jnp.sqrt(imgs[len(imgs) // 2]), HR, "bilinear"))
    print(f"FPM phase corr {corr(np.angle(rec), np.angle(hr)):+.3f} | "
          f"amp corr {corr(np.abs(rec), np.abs(hr)):+.3f} | "
          f"single-LED corr {corr(single, np.abs(hr)):+.3f}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 4, figsize=(14, 3.6))
    panels = [(np.angle(hr), "true phase", "twilight"),
              (single, "single LED (objective-limited)", "gray"),
              (np.angle(rec), "FPM reconstructed phase", "twilight"),
              (np.abs(rec), "FPM reconstructed amplitude", "gray")]
    for a, (img, t, cm) in zip(ax, panels):
        a.imshow(img, cmap=cm); a.axis("off"); a.set_title(t, fontsize=10)
    fig.suptitle(f"Fourier Ptychography: {imgs.shape[0]} low-res LED images "
                 f"-> super-resolved complex object (JAX/Optax, {LR}->{HR})", fontsize=11)
    fig.tight_layout(); fig.savefig("fpm_demo.png", dpi=110)
    print("wrote fpm_demo.png")


if __name__ == "__main__":
    main()
