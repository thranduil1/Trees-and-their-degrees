import numpy as np
from dimension_one.making_tree import make_family_one, tree_at_one

def edge_slice_intersection(
    start,
    end,
    *,
    slice_time=0.0,
    time_coordinate=-1,
    tolerance=1e-12,
):
    """
    Intersect segment [start,end] with time = slice_time.
    Return transverse coordinates or None.
    """
    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)

    start_time = start[time_coordinate]
    end_time = end[time_coordinate]

    denominator = end_time - start_time

    if abs(denominator) < tolerance:
        return None

    parameter = (
        (slice_time - start_time)
        / denominator
    )

    if parameter < -tolerance:
        return None

    if parameter > 1.0 + tolerance:
        return None

    point = start + parameter * (
        end - start
    )

    transverse = np.delete(
        point,
        time_coordinate,
    )

    if np.any(transverse < -tolerance):
        return None

    if np.any(transverse > 1.0 + tolerance):
        return None

    return np.clip(
        transverse,
        0.0,
        1.0,
    )

def closest_intersection_to_center(
    intersections,
    center=None,
    tolerance=1e-12,
):
    """
    Given a list of intersection points in [0,1]^(N-1),
    return the unique one closest to the center.

    If there is a tie, return None (basepoint).
    """
    if not intersections:
        return None

    if center is None:
        center = np.full(
            len(intersections[0]),
            0.5,
        )

    distances = [
        np.linalg.norm(
            point - center
        )
        for point in intersections
    ]

    min_distance = min(distances)

    closest = [
        point
        for point, dist in zip(
            intersections,
            distances,
        )
        if abs(dist - min_distance) < tolerance
    ]

    if len(closest) != 1:
        return None

    return closest[0]

def one_dimensional_detector(
    tree,
    *,
    slice_time=0.0,
    tolerance=1e-12,
):
    """
    Detector for the one-dimensional family.

    Returns:
        a point in [0,1]^(N-1), or
        None for the quotient basepoint.
    """
    vertices = tree["vertices"]
    time_coordinate = tree.get(
        "time_coordinate",
        -1,
    )

    rootward_intersection = None
    leafward_intersections = []

    for left_name, right_name in tree["edges"]:
        intersection = edge_slice_intersection(
            vertices[left_name],
            vertices[right_name],
            slice_time=slice_time,
            time_coordinate=time_coordinate,
            tolerance=tolerance,
        )

        if intersection is None:
            continue

        vertex_type = tree["vertex_type"]

        if (
            vertex_type.get(left_name) == "a"
            and vertex_type.get(right_name) == "root"
        ) or (
            vertex_type.get(left_name) == "root"
            and vertex_type.get(right_name) == "a"
        ):
            rootward_intersection = intersection

        elif (
            vertex_type.get(left_name) == "leaf"
            or vertex_type.get(right_name) == "leaf"
        ):
            leafward_intersections.append(
                intersection
            )

    # Case 1: rootward edge intersects the slice.
    if rootward_intersection is not None:
        return rootward_intersection

    # Case 2: all leafward edges intersect the slice.
    num_leafward = len([
        name
        for name, kind in tree["vertex_type"].items()
        if kind == "leaf"
    ])

    if len(leafward_intersections) == num_leafward:
        return closest_intersection_to_center(
            leafward_intersections,
            tolerance=tolerance,
        )

    # Otherwise: no valid output.
    return None

def make_slice_detector_map(
    family,
):
    """
    Return the map

        [0,1]^(N-1) -> [0,1]^(N-1) / boundary.

    Output None represents the quotient basepoint.
    """
    def detector_map(x):
        tree = tree_at_one(
            family,
            x,
        )

        return one_dimensional_detector(
            tree,
            slice_time=family["slice_time"],
        )

    return detector_map
