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
    N,
    *,
    a,
    angle_degrees=30.0,
    edge_length=1.0,
):
    """
    Family of one-junction trees for input p in [0,1]^N,
    with an a-ary junction (1 root edge, a leaf edges).

    Transverse dimension d = N - 1.
    Time coordinate index = d.

    Parameters
    ----------
    N : int
        Dimension of the input cube [0,1]^N.
    a : int
        Number of leaf edges (branching number).
    angle_degrees : float
        Angle of leaf edges relative to the time axis.
    edge_length : float
        Length of root and leaf edges in space-time.

    Returns
    -------
    family : dict
    """
    if N < 2:
        raise ValueError("N must be at least 2 (so that d = N-1 >= 1).")
    if a < 1:
        raise ValueError("a must be at least 1.")

    d = N - 1  # transverse dimension
    time_coordinate = d

    angle = np.deg2rad(angle_degrees)

    # Root direction: purely in +time, no transverse component
    root_direction = np.zeros(N, dtype=float)
    root_direction[time_coordinate] = 1.0

    # Leaf directions: a symmetric directions in transverse space,
    # all with the same negative time component.
    #
    # We embed them in a 2D subspace of the transverse space for simplicity:
    # angles theta_k = 2π k / a, k = 0,...,a-1.
    leaf_directions = []
    for k in range(a):
        theta = 2.0 * np.pi * k / a
        transverse = np.zeros(d, dtype=float)
        if d >= 2:
            # Put the angular pattern in the first two transverse coords
            transverse[0] = np.sin(angle) * np.cos(theta)
            transverse[1] = np.sin(angle) * np.sin(theta)
        elif d == 1:
            # In 1D transverse space, alternate left/right
            transverse[0] = np.sin(angle) * (1 if k % 2 == 0 else -1)
        time_component = -np.cos(angle)

        direction = np.concatenate([transverse, [time_component]])
        leaf_directions.append(direction)

    leaf_directions = np.array(leaf_directions, dtype=float)

    return {
        "N": N,
        "transverse_dimension": d,
        "time_coordinate": time_coordinate,
        "edge_length": float(edge_length),
        "root_direction": root_direction,
        "leaf_directions": leaf_directions,
        "a": int(a),
        "angle_degrees": float(angle_degrees),
    }

def tree_at_one(family, p):
    """
    Build a one-junction tree for input p in [0,1]^N,
    with an a-ary junction (1 root, a leaves).

    Junction transverse coordinates = first N-1 coords of p.
    Junction time = last coord of p.

    Parameters
    ----------
    family : dict
        From make_family.
    p : array_like
        Point in [0,1]^N.

    Returns
    -------
    tree : dict
    """
    N = family["N"]
    d = family["transverse_dimension"]
    time_coordinate = family["time_coordinate"]
    edge_length = family["edge_length"]
    a = family["a"]

    root_direction = family["root_direction"]
    leaf_directions = family["leaf_directions"]  # shape (a, N)

    p = np.asarray(p, dtype=float)

    if p.shape != (N,):
        raise ValueError(f"p must have shape ({N},), got {p.shape}")

    # Junction = (transverse, time) = (p_1,...,p_{N-1}, p_N)
    junction = p.copy()

    # Root vertex
    root = junction + edge_length * root_direction

    # Leaf vertices
    leaves = [
        junction + edge_length * v for v in leaf_directions
    ]

    vertices = {
        "a0": junction,
        "root": root,
    }
    for i, leaf in enumerate(leaves):
        vertices[f"leaf{i}"] = leaf

    edges = [("a0", "root")] + [
        ("a0", f"leaf{i}") for i in range(a)
    ]

    return {
        "N": N,
        "transverse_dimension": d,
        "time_coordinate": time_coordinate,
        "a": a,
        "vertices": vertices,
        "edges": edges,
    }
    
