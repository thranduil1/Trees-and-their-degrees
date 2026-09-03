import numpy as np

# Distinguish a "time direction" in which the leaves are progressing from leaves to root.
# The following two functions either distinguish that special dimension or integrate it back in.

def split_space_time(point):
    point = np.asarray(point, dtype=float)

    return point[:-1], point[-1]

def join_space_time(transverse, time):
    transverse = np.asarray(
        transverse,
        dtype=float,
    )

    return np.concatenate([
        transverse,
        [float(time)],
    ])

import numpy as np


def make_family_one(
    ambient_dimension,
    *,
    junction_time=0.5,
    angle_degrees=30.0,
):
    """
    Family of trees in [0,1]^N.

    Parameters
    ----------
    ambient_dimension:
        N, the dimension of the ambient cube.

    junction_time:
        Fixed time-coordinate of the unique a-junction.

    angle_degrees:
        Fixed angle used to define the incident edges.

    The detector slice is always time=0, using the final
    ambient coordinate as time.
    """
    if ambient_dimension < 2:
        raise ValueError(
            "ambient_dimension must be at least 2"
        )

    if not 0.0 < junction_time < 1.0:
        raise ValueError(
            "junction_time must lie strictly between 0 and 1"
        )

    return {
        "ambient_dimension": ambient_dimension,
        "transverse_dimension": ambient_dimension - 1,
        "junction_time": float(junction_time),
        "angle_radians": np.deg2rad(
            angle_degrees
        ),
        "slice_time": 0.0,
    }

def a_junction_directions(
    family,
):
    """
    Two fixed unit directions at the a-junction.

    The first edge points toward decreasing time and meets
    the detector slice. The second points toward increasing time.
    """
    transverse_dimension = family[
        "transverse_dimension"
    ]
    angle = family["angle_radians"]

    down = np.zeros(
        transverse_dimension + 1
    )
    up = np.zeros(
        transverse_dimension + 1
    )

    # First transverse direction plus final time direction.
    down[0] = np.sin(angle)
    down[-1] = -np.cos(angle)

    up[0] = -np.sin(angle)
    up[-1] = np.cos(angle)

    return down, up

def ray_to_cube_boundary(
    start,
    direction,
    tolerance=1e-12,
):
    """
    Follow start + s*direction until it first reaches
    the boundary of [0,1]^N.
    """
    start = np.asarray(start, dtype=float)
    direction = np.asarray(
        direction,
        dtype=float,
    )

    candidates = []

    for coordinate in range(len(start)):
        if direction[coordinate] > tolerance:
            candidates.append(
                (1.0 - start[coordinate])
                / direction[coordinate]
            )

        elif direction[coordinate] < -tolerance:
            candidates.append(
                -start[coordinate]
                / direction[coordinate]
            )

    positive = [
        value
        for value in candidates
        if value > tolerance
    ]

    if not positive:
        raise RuntimeError(
            "Ray does not meet cube boundary."
        )

    scale = min(positive)

    return start + scale * direction

def tree_at_one(
    family,
    x,
):
    """
    Construct the tree corresponding to transverse parameter x.

    The tree contains one a-junction at (x, junction_time),
    one leafward endpoint, and one rootward endpoint.
    """
    x = np.asarray(x, dtype=float)

    transverse_dimension = family[
        "transverse_dimension"
    ]

    if x.shape != (transverse_dimension,):
        raise ValueError(
            "x has the wrong transverse dimension."
        )

    if np.any(x < 0.0) or np.any(x > 1.0):
        raise ValueError(
            "x must lie in [0,1]^(N-1)."
        )

    junction = join_space_time(
        x,
        family["junction_time"],
    )

    downward, upward = a_junction_directions(
        family
    )

    leaf = ray_to_cube_boundary(
        junction,
        downward,
    )

    root = ray_to_cube_boundary(
        junction,
        upward,
    )

    return {
        "vertices": {
            "leaf": leaf,
            "a0": junction,
            "root": root,
        },
        "vertex_type": {
            "leaf": "leaf",
            "a0": "a",
            "root": "root",
        },
        "edges": [
            ("leaf", "a0"),
            ("a0", "root"),
        ],
        "time_coordinate": (
            family["ambient_dimension"] - 1
        ),
    }

