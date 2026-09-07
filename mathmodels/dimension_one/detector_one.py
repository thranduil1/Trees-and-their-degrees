import numpy as np
from dimension_one.making_tree import make_family_one, tree_at_one

def edge_slice_intersection(
    start,
    end,
    *,
    slice_time=0.0,
    time_coordinate,
):
    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)

    start_time = start[time_coordinate]
    end_time = end[time_coordinate]

    if (start_time - slice_time) * (end_time - slice_time) > 0:
        return None

    denom = end_time - start_time
    if abs(denom) < 1e-12:
        if abs(start_time - slice_time) < 1e-12:
            return start.copy()
        return None

    t = (slice_time - start_time) / denom
    if not (0.0 <= t <= 1.0):
        return None

    return start + t * (end - start)

def one_dimensional_detector(family, p, *, slice_time):
    """
    Detector map for input p in [0,1]^N.

    Intersects the tree with the time slice t = slice_time.
    Returns a point in [0,1]^{N-1} (transverse space) or None (basepoint).
    """
    N = family["N"]
    d = family["transverse_dimension"]
    time_coordinate = family["time_coordinate"]

    tree = tree_at_one(family, p)
    vertices = tree["vertices"]
    edges = tree["edges"]

    rootward_edge = ("a0", "root")
    leafward_edges = [
        e for e in edges if e != rootward_edge
    ]

    def intersect_edge(left_name, right_name):
        return edge_slice_intersection(
            vertices[left_name],
            vertices[right_name],
            slice_time=slice_time,
            time_coordinate=time_coordinate,
        )

    rootward_intersection = intersect_edge(*rootward_edge)

    if rootward_intersection is not None:
        return rootward_intersection[:d]

    leaf_intersections = []
    for e in leafward_edges:
        pt = intersect_edge(*e)
        if pt is None:
            return None  # not all leafward edges hit the slice
        leaf_intersections.append(pt)

    center = np.full(d, 0.5)

    def dist_to_center(pt):
        return np.linalg.norm(pt[:d] - center)

    closest = min(leaf_intersections, key=dist_to_center)
    return closest[:d]

def make_slice_detector_map(family, slice_time=0.5):
    def detector_map(p):
        return one_dimensional_detector(family, p, slice_time = slice_time)
    return detector_map
