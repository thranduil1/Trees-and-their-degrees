

import numpy as np
import matplotlib.pyplot as plt

from simple_junction.making_tree_simple import make_family_one
from simple_junction.detector_simple import make_slice_detector_map

# This function computes a grid of detector outputs and plots a heat map.
# The heat map shows the Euclidean distance from each detector output to the center of the transverse cube. 
# Points sent to the basepoint (None) are shown in white.

## Detector heat map

"""
Heat map for N=3 detector, with one coordinate fixed.

Colors each (p_i, p_j) by distance of the detector output to the
boundary of the transverse square [0,1]^2. Basepoint outputs are
shown as NaN (white/transparent).
"""

def distance_to_boundary(q):
    """
    Distance from q in [0,1]^2 to the boundary of the square.

    Returns a scalar in [0, 0.5].
    """
    if q is None:
        return np.nan
    q = np.asarray(q, dtype=float)
    if q.shape != (2,):
        raise ValueError("q must be in R^2")
    return float(
        min(
            q[0],
            1.0 - q[0],
            q[1],
            1.0 - q[1],
        )
    )


def detector_heatmap_N3(
    family,
    *,
    fixed_coord="p3",
    fixed_value=0.5,
    samples=101,
    cmap="viridis",
    slice_time = 0.5
):
    """
    Plot a 2D heat map of the detector for N=3.

    Parameters
    ----------
    family : dict
        From make_family with N=3.
    fixed_coord : {"p1", "p2", "p3"}
        Which input coordinate to fix.
    fixed_value : float
        Value at which to fix that coordinate (in [0,1]).
    samples : int
        Number of samples per axis.
    cmap : str
        Matplotlib colormap.

    Returns
    -------
    fig, ax, grid, values
    """

    if family["N"] != 3:
        raise ValueError("This heat map is for N=3 only.")

    d = family["transverse_dimension"]
    if d != 2:
        raise ValueError("Expected transverse_dimension=2 for N=3.")

    detector_map = make_slice_detector_map(family, slice_time)

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

    for i, (x, y) in enumerate(grid):
        if fixed_coord == "p1":
            p = np.array([fixed_value, x, y])
        elif fixed_coord == "p2":
            p = np.array([x, fixed_value, y])
        elif fixed_coord == "p3":
            p = np.array([x, y, fixed_value])
        else:
            raise ValueError("fixed_coord must be 'p1', 'p2', or 'p3'.")

        out = detector_map(p)
        values[i] = out

    dists = np.array(
        [distance_to_boundary(v) for v in values],
        dtype=float,
    )

    Z = dists.reshape(X.shape)

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
        label="Distance to boundary",
    )

    # Label axes according to which coordinates are varying
    if fixed_coord == "p1":
        ax.set_xlabel("p2")
        ax.set_ylabel("p3")
        title = f"N=3 detector (p1={fixed_value:.2f} fixed)"
    elif fixed_coord == "p2":
        ax.set_xlabel("p1")
        ax.set_ylabel("p3")
        title = f"N=3 detector (p2={fixed_value:.2f} fixed)"
    else:  # p3
        ax.set_xlabel("p1")
        ax.set_ylabel("p2")
        title = f"N=3 detector (p3={fixed_value:.2f} fixed)"

    ax.set_title(title)

    plt.tight_layout()

    return fig, ax, grid, values
