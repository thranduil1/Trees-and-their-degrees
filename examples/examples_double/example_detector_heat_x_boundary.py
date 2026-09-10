# We are plotting the detector map on the boundary of our cube C^{N+1}.
# Recall that an element in the cube is a pair (x,t) with x in C^N and t in I.
# On the boundary of C^{N+1}, we have either that x is on the boundary of C^N or t=0 or t=1.
# Right now we are looking at values on the boundary of C^N.

import matplotlib.pyplot as plt
import numpy as np
from double_junction.Tree_building import make_tree_family
from detector_heat_map import plot_detector_slice


family = make_tree_family(a=3, b=2, N=3)


fig, axes = plt.subplots(
    1, 2,
    figsize=(14, 6),
    constrained_layout=True,
)


# Left: x[0] = 0
_, _, distances0, active0 = plot_detector_slice(
    family,
    target_type="a",
    fixed_axis=0,
    fixed_value=0.0,
    varying_axis=1,
    other_x_value=0.5,
    resolution=150,
    title="a-detector: x[0]=0, vary x[1] and t",
)
# Overwrite the axes created inside plot_detector_slice with our own:
axes[0].imshow(
    distances0,
    origin="lower",
    extent=[0.0, 1.0, 0.0, 1.0],
    aspect="auto",
    cmap="magma",
    vmin=0.0,
    vmax=np.pi,
    interpolation="nearest",
)
if np.any(active0) and not np.all(active0):
    x_vals = np.linspace(0.0, 1.0, distances0.shape[1])
    t_vals = np.linspace(0.0, 1.0, distances0.shape[0])
    axes[0].contour(
        x_vals, t_vals,
        active0.astype(float),
        levels=[0.5],
        colors="white",
        linewidths=0.8,
    )
axes[0].set_xlabel("parameter x[1]")
axes[0].set_ylabel("tree-shape parameter t")
axes[0].set_title("a-detector: x[0]=0, vary x[1] and t")


# Right: x[0] = 1
_, _, distances1, active1 = plot_detector_slice(
    family,
    target_type="a",
    fixed_axis=0,
    fixed_value=1.0,
    varying_axis=1,
    other_x_value=0.5,
    resolution=150,
    title="a-detector: x[0]=1, vary x[1] and t",
)
axes[1].imshow(
    distances1,
    origin="lower",
    extent=[0.0, 1.0, 0.0, 1.0],
    aspect="auto",
    cmap="magma",
    vmin=0.0,
    vmax=np.pi,
    interpolation="nearest",
)
if np.any(active1) and not np.all(active1):
    x_vals = np.linspace(0.0, 1.0, distances1.shape[1])
    t_vals = np.linspace(0.0, 1.0, distances1.shape[0])
    axes[1].contour(
        x_vals, t_vals,
        active1.astype(float),
        levels=[0.5],
        colors="white",
        linewidths=0.8,
    )
axes[1].set_xlabel("parameter x[1]")
axes[1].set_ylabel("tree-shape parameter t")
axes[1].set_title("a-detector: x[0]=1, vary x[1] and t")


# Shared colorbar
sm = plt.cm.ScalarMappable(
    cmap="magma",
    norm=plt.Normalize(vmin=0.0, vmax=np.pi),
)
cbar = fig.colorbar(sm, ax=axes, shrink=0.8)
cbar.set_label("Spherical distance to quotient basepoint")


plt.savefig(
    "results/boundary_x.png",
    dpi=200,
    bbox_inches="tight",
)
plt.show()