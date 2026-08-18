"""End-to-end demo on a synthetic map: forward-model two shift maps from a
known strain field and a known charge patch, add noise, invert, and plot.
Uses the synthetic (non-physical) demo coefficients."""
import numpy as np

from ramansep import SeparationModel, synthetic_demo

model = SeparationModel(synthetic_demo())

# ground truth: a strain gradient and a charged edge stripe
y, x = np.mgrid[0:128, 0:128]
strain_true = 0.05 * (x / 128.0)          # smooth tension gradient
density_true = np.where(x > 110, 2.0, 0.0)  # charge at the "edge"

dw1, dw2 = model.forward(strain_true, density_true)
rng = np.random.default_rng(42)
dw1 += rng.normal(0.0, 0.05, dw1.shape)
dw2 += rng.normal(0.0, 0.05, dw2.shape)

result = model.invert(dw1, dw2, sigma1=0.05, sigma2=0.05)
print("condition number:", result.condition_number)
print("mean absolute strain error:", np.abs(result.strain - strain_true).mean())
print("mean absolute density error:", np.abs(result.density - density_true).mean())

try:
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(8, 7), constrained_layout=True)
    for ax, (field, title) in zip(axes.flat, [
        (strain_true, "strain (truth)"), (result.strain, "strain (recovered)"),
        (density_true, "density (truth)"), (result.density, "density (recovered)"),
    ]):
        im = ax.imshow(field); ax.set_title(title); fig.colorbar(im, ax=ax)
    fig.savefig("synthetic_map.png", dpi=150)
    print("wrote synthetic_map.png")
except ImportError:
    print("matplotlib not installed; skipped the plot")
