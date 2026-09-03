

import numpy as np
import matplotlib.pyplot as plt

from dimension_one.making_tree import make_family_one
from dimension_one.detector_one import make_slice_detector_map

# This function computes a grid of detector outputs and plots a heat map.
# The heat map shows the Euclidean distance from each detector output to the center of the transverse cube. 
# Points sent to the basepoint (None) are shown in white.

def detector_heatmap(
    family,
    *,
    samples=101,
    cmap="viridis",
):
    transverse_dimension = family["transverse_dimension"]

    if transverse_dimension != 2:
        raise ValueError(
            "Heat map currently implemented only for "
            "transverse dimension 2."
        )

    detector_map = make_slice_detector_map(
        family,
    )

    xs = np.linspace(0.0, 1.0, samples)
    ys = np.linspace(0.0, 1.0, samples)

    X, Y = np.meshgrid(xs, ys)

    grid = np.stack(
        [X.ravel(), Y.ravel()],
        axis=-1,
    )

    values = np.empty(
        grid.shape[0],
        dtype=object,
    )

    for i, point in enumerate(grid):
        values[i] = detector_map(point)

    center = np.full(transverse_dimension, 0.5)

    distances = np.empty(
        grid.shape[0],
        dtype=float,
    )

    for i, val in enumerate(values):
        if val is None:
            distances[i] = np.nan
        else:
            distances[i] = np.linalg.norm(
                np.asarray(val) - center
            )

    Z = distances.reshape(X.shape)

    fig, ax = plt.subplots(
        figsize=(6, 5),
    )

    im = ax.imshow(
        Z,
        origin="lower",
        extent=[0.0, 1.0, 0.0, 1.0],
        cmap=cmap,
        aspect="auto",
    )

    cbar = fig.colorbar(
        im,
        ax=ax,
        label="Distance to center",
    )

    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_title("Detector heat map")

    plt.tight_layout()

    return fig, ax, grid, values


def detector_vector_field(
    family,
    *,
    samples=21,
    arrow_scale=0.15,
):
    """
    Plot the detector as a vector field: arrows from input x
    to output f(x). Basepoint outputs are omitted.
    """
    transverse_dimension = family["transverse_dimension"]

    if transverse_dimension != 2:
        raise ValueError(
            "Vector field currently implemented only for "
            "transverse dimension 2."
        )

    detector_map = make_slice_detector_map(
        family,
    )

    xs = np.linspace(0.0, 1.0, samples)
    ys = np.linspace(0.0, 1.0, samples)

    X, Y = np.meshgrid(xs, ys)

    U = np.full_like(X, np.nan)
    V = np.full_like(Y, np.nan)

    for i in range(samples):
        for j in range(samples):
            x = np.array([xs[j], ys[i]])

            out = detector_map(x)

            if out is None:
                continue

            out = np.asarray(out, dtype=float)

            U[i, j] = out[0] - x[0]
            V[i, j] = out[1] - x[1]

    fig, ax = plt.subplots(
        figsize=(6, 5),
    )

    ax.quiver(
        X, Y, U, V,
        angles="xy",
        scale_units="xy",
        scale=1.0 / arrow_scale,
        width=0.004,
        color="blue",
    )

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_aspect("equal")

    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_title("Detector vector field")

    plt.tight_layout()

    return fig, ax, X, Y, U, V
