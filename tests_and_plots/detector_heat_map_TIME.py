# Now we are plotting the detector map on a 2D slice on the boundary of the cube.
# But only on the faces which correspond to t=0 or t=1.

import numpy as np
import matplotlib.pyplot as plt
from Tree_building import *
from Detector_maps import detect_a, detect_b
"""
    Plot detector distance to the quotient basepoint on t=0 or t=1.

    For N=3, two coordinates of x are varied and the remaining
    coordinate is fixed.

    Parameters
    ----------
    family:
        Tree-family dictionary from make_tree_family(...).

    target_type:
        "a" or "b".

    t_value:
        Exactly 0.0 or 1.0.

    varying_axes:
        Two x-coordinate indices plotted horizontally and vertically.

    fixed_x_value:
        Value assigned to all x-coordinates not in varying_axes.

    resolution:
        Number of grid points in each plotted direction.

    detector_kwargs:
        Optional keyword arguments passed to detect_a/detect_b.

    Returns
    -------
    fig, ax, distances, active
"""

family = make_tree_family(a=3,b=2,N=3,)

def plot_detector_time_face(
    family,
    target_type,
    *,
    t_value,
    varying_axes=(0, 1),
    fixed_x_value=0.5,
    resolution=150,
    detector_kwargs=None,
    title=None,
):

    N = family["N"]

    if N != 3:
        raise ValueError(
            "This plotting function is intended for N=3."
        )

    if target_type not in {"a", "b"}:
        raise ValueError(
            "target_type must be 'a' or 'b'."
        )

    if t_value not in {0.0, 1.0}:
        raise ValueError(
            "t_value must be exactly 0.0 or 1.0."
        )

    varying_axes = tuple(varying_axes)

    if len(varying_axes) != 2:
        raise ValueError(
            "varying_axes must contain exactly two indices."
        )

    if len(set(varying_axes)) != 2:
        raise ValueError(
            "The two varying axes must be different."
        )

    if any(axis not in {0, 1, 2} for axis in varying_axes):
        raise ValueError(
            "For N=3, axes must be 0, 1, or 2."
        )

    if detector_kwargs is None:
        detector_kwargs = {}

    detector = detect_a if target_type == "a" else detect_b

    horizontal_axis, vertical_axis = varying_axes

    horizontal_values = np.linspace(
        0.0,
        1.0,
        resolution,
    )

    vertical_values = np.linspace(
        0.0,
        1.0,
        resolution,
    )

    distances = np.zeros(
        (resolution, resolution),
        dtype=float,
    )

    active = np.zeros(
        (resolution, resolution),
        dtype=bool,
    )

    for row, vertical_value in enumerate(vertical_values):
        for column, horizontal_value in enumerate(horizontal_values):
            x = np.full(
                N,
                fixed_x_value,
                dtype=float,
            )

            x[horizontal_axis] = horizontal_value
            x[vertical_axis] = vertical_value

            tree = tree_at(
                family,
                x,
                t_value,
            )

            q = detector(
                tree,
                **detector_kwargs,
            )

            if q is None:
                distances[row, column] = 0.0
                active[row, column] = False
            else:
                q = np.asarray(q, dtype=float)

                # Distance to the boundary of the target cube.
                # This is zero at the quotient basepoint and
                # largest near the target-cube center.
                distances[row, column] = np.min(
                    np.minimum(q, 1.0 - q)
                )

                active[row, column] = True

    fig, ax = plt.subplots(figsize=(8, 7))

    image = ax.imshow(
        distances,
        origin="lower",
        extent=[
            0.0,
            1.0,
            0.0,
            1.0,
        ],
        aspect="equal",
        cmap="magma",
        interpolation="nearest",
    )

    if np.any(active) and not np.all(active):
        ax.contour(
            horizontal_values,
            vertical_values,
            active.astype(float),
            levels=[0.5],
            colors="cyan",
            linewidths=0.9,
        )

    colorbar = fig.colorbar(
        image,
        ax=ax,
    )

    colorbar.set_label(
        "Distance to quotient basepoint"
    )

    ax.set_xlabel(
        f"$x_{horizontal_axis}$"
    )

    ax.set_ylabel(
        f"$x_{vertical_axis}$"
    )

    if title is None:
        title = (
            f"{target_type}-detector on t={t_value:g}; "
            f"varying x[{horizontal_axis}], x[{vertical_axis}]"
        )

    ax.set_title(title)

    plt.tight_layout()

    return fig, ax, distances, active
