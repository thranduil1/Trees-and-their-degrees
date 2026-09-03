def edge_slice_intersection(
    start,
    end,
    *,
    slice_time=0.0,
    time_coordinate=-1,
    tolerance=1e-12,
):
    """
    Intersect the segment [start,end] with the hyperplane
    p[time_coordinate] = slice_time.

    Returns:
        transverse coordinates in [0,1]^(N-1), or None.
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

def one_dimensional_detector(
    tree,
    *,
    slice_time=0.0,
    tolerance=1e-12,
):
    """
    Detect the unique edge intersecting the time-zero slice.

    Returns:
        an element of [0,1]^(N-1), or
        None for the quotient basepoint.
    """
    vertices = tree["vertices"]
    time_coordinate = tree.get(
        "time_coordinate",
        -1,
    )

    intersections = []

    for left_name, right_name in tree["edges"]:
        intersection = edge_slice_intersection(
            vertices[left_name],
            vertices[right_name],
            slice_time=slice_time,
            time_coordinate=time_coordinate,
            tolerance=tolerance,
        )

        if intersection is not None:
            intersections.append({
                "edge": (left_name, right_name),
                "point": intersection,
            })

    if len(intersections) != 1:
        return None

    return intersections[0]["point"]

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
