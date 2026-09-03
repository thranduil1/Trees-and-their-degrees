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


import numpy as np


def make_family_one(
    ambient_dimension,
    *,
    junction_time=0.5,
    angle_degrees=30.0,
    num_leafward_edges=None,
):
    """
    One-dimensional detector family.

    Parameters
    ----------
    ambient_dimension:
        N, the dimension of the ambient cube [0,1]^N.

    junction_time:
        Fixed time-coordinate of the unique a-junction.

    angle_degrees:
        Fixed angle for all edges at the junction.

    num_leafward_edges:
        Number of leafward edges. If None, set to ambient_dimension.

    The detector slice is always time=0.
    """
    if ambient_dimension < 2:
        raise ValueError(
            "ambient_dimension must be at least 2"
        )

    if not 0.0 < junction_time < 1.0:
        raise ValueError(
            "junction_time must lie strictly between 0 and 1"
        )

    if num_leafward_edges is None:
        num_leafward_edges = ambient_dimension

    if num_leafward_edges < 1:
        raise ValueError(
            "num_leafward_edges must be positive"
        )

    return {
        "ambient_dimension": ambient_dimension,
        "transverse_dimension": ambient_dimension - 1,
        "junction_time": float(junction_time),
        "angle_radians": np.deg2rad(
            angle_degrees
        ),
        "num_leafward_edges": int(
            num_leafward_edges
        ),
        "slice_time": 0.0,
    }

def a_junction_directions(
    family,
):
    """
    Return directions of all edges at the a-junction.

    Returns
    -------
    rootward:
        Unit direction pointing toward increasing time.
    leafward:
        List of unit directions pointing toward decreasing time.
    """
    transverse_dimension = family[
        "transverse_dimension"
    ]
    angle = family["angle_radians"]
    num_leafward = family["num_leafward_edges"]

    # Rootward: purely in time direction.
    rootward = np.zeros(
        transverse_dimension + 1
    )
    rootward[-1] = 1.0

    # Leafward: one per transverse direction.
    leafward = []

    for i in range(num_leafward):
        direction = np.zeros(
            transverse_dimension + 1
        )

        if i < transverse_dimension:
            direction[i] = np.sin(angle)

        direction[-1] = -np.cos(angle)

        leafward.append(direction)

    return rootward, leafward
    
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
    Construct the tree for transverse parameter x.

    The tree has:
    - one a-junction at (x, junction_time);
    - one rootward edge;
    - num_leafward_edges leafward edges.
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

    junction = np.concatenate([
        x,
        [family["junction_time"]],
    ])

    rootward, leafward = a_junction_directions(
        family
    )

    root = ray_to_cube_boundary(
        junction,
        rootward,
    )

    leaves = [
        ray_to_cube_boundary(
            junction,
            direction,
        )
        for direction in leafward
    ]

    vertices = {
        "a0": junction,
        "root": root,
    }

    vertex_type = {
        "a0": "a",
        "root": "root",
    }

    edges = [
        ("a0", "root"),
    ]

    for i, leaf in enumerate(leaves):
        name = f"leaf{i}"
        vertices[name] = leaf
        vertex_type[name] = "leaf"
        edges.append(
            (name, "a0")
        )

    return {
        "vertices": vertices,
        "vertex_type": vertex_type,
        "edges": edges,
        "time_coordinate": transverse_dimension,
    }
