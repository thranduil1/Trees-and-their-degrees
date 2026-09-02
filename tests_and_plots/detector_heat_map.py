# Plotting a heat map of the detector on 2D slices to check what it looks like + continuity.
# Obviously this is not enough to check full continuity but it is a good first sanity check :)

import numpy as np
import matplotlib.pyplot as plt

# What we are actually plotting is the distance from a quotient-cube point to the basepoint. 
# We use the quotient_cube_to_sphere representation: C^N / boundary(C^N) ~= S^N.
# The basepoint is the north pole, so the returned value is its spherical geodesic distance from q.
# Important note: this is just for plotting so it doesn't matter but we will have to think later about the model we want to use for our quotient: 
# C^N / boundary(C^N) or S^N ? This will be key in degree computations.

def detector_distance_to_basepoint(q, N):

    sphere_point = quotient_cube_to_sphere(q, N)

    north_coordinate = np.clip(
        sphere_point[-1],
        -1.0,
        1.0,
    )

    return float(np.arccos(north_coordinate))

"""
    Plot detector distance-to-basepoint on a 2D boundary slice.

    The plotted slice is:

        x[fixed_axis] = fixed_value,
        x[varying_axis] varies horizontally in [0,1],
        t varies vertically in [0,1],
        all remaining x-coordinates = other_x_value.

    Since fixed_value is 0 or 1, this is a slice inside
    boundary(C^(N+1)).

    Parameters
    ----------
    target_type:
        Either "a" or "b".

    fixed_axis:
        Which x-coordinate is fixed to 0 or 1.

    fixed_value:
        Must be 0.0 or 1.0, to remain on the parameter-cube
        boundary.

    varying_axis:
        Which remaining x-coordinate is displayed horizontally.

    other_x_value:
        Fixed value for all unused x-coordinates.

    resolution:
        Number of sample points in each plotted direction.
    """

def plot_detector_slice(
    family,
    target_type,
    *,
    fixed_axis=0,
    fixed_value=0.0,
    varying_axis=1,
    other_x_value=0.5,
    resolution=120,
    title=None,
):

    N = family["N"]

    if target_type not in {"a", "b"}:
        raise ValueError(
            "target_type must be 'a' or 'b'."
        )

    if fixed_axis not in range(N):
        raise ValueError(
            "fixed_axis must be an x-coordinate index."
        )

    if varying_axis not in range(N):
        raise ValueError(
            "varying_axis must be an x-coordinate index."
        )

    if fixed_axis == varying_axis:
        raise ValueError(
            "fixed_axis and varying_axis must differ."
        )

    if fixed_value not in {0.0, 1.0}:
        raise ValueError(
            "fixed_value must be 0.0 or 1.0."
        )

    if target_type == "a":
        detector = detect_a
    else:
        detector = detect_b

    x_values = np.linspace(0.0, 1.0, resolution)
    t_values = np.linspace(0.0, 1.0, resolution)

    distances = np.zeros(
        (resolution, resolution),
        dtype=float,
    )

    active = np.zeros(
        (resolution, resolution),
        dtype=bool,
    )

    for row, t in enumerate(t_values):
        for column, varying_value in enumerate(x_values):
            x = np.full(N, other_x_value)

            x[fixed_axis] = fixed_value
            x[varying_axis] = varying_value

            tree = tree_at(
                family,
                x,
                float(t),
            )

            q = detector(tree)

            distances[row, column] = (
                detector_distance_to_basepoint(q, N)
            )

            active[row, column] = q is not None

    fig, ax = plt.subplots(figsize=(8, 6))

    image = ax.imshow(
        distances,
        origin="lower",
        extent=[0.0, 1.0, 0.0, 1.0],
        aspect="auto",
        cmap="magma",
        vmin=0.0,
        vmax=np.pi,
        interpolation="nearest",
    )

    # White curve: detector changes between basepoint and active.
    if np.any(active) and not np.all(active):
        ax.contour(
            x_values,
            t_values,
            active.astype(float),
            levels=[0.5],
            colors="white",
            linewidths=0.8,
        )

    colorbar = fig.colorbar(
        image,
        ax=ax,
    )

    colorbar.set_label(
        "Spherical distance to quotient basepoint"
    )

    ax.set_xlabel(
        f"parameter x[{varying_axis}]"
    )

    ax.set_ylabel("tree-shape parameter t")

    if title is None:
        title = (
            f"{target_type}-detector on "
            f"x[{fixed_axis}] = {fixed_value}"
        )

    ax.set_title(title)

    plt.tight_layout()

    return fig, ax, distances, active
