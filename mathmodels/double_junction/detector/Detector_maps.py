## Map that detects what is at the center of a tree embedding ##
## For more info on what we are trying to do here, look at the mathcontext folder.

import numpy as np

# L-infinity geometry: we want to determine the closest junction to the center for the L-infinity distance.

def linf_distance(p, q):
    """
    L-infinity distance between two points.
    """
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)

    if p.shape != q.shape:
        raise ValueError("Points must have the same shape.")

    return float(np.max(np.abs(p - q)))


def linf_distance_point_to_segment(point, start, end):
    """
    Exact L-infinity distance from a point to a closed affine segment.

    Computes:

        min_{0 <= s <= 1}
        || point - ((1-s) * start + s * end) ||_infinity.

    The objective is convex and piecewise linear in s. Its minimum
    occurs either at s = 0, s = 1, or where two affine pieces meet.
    """
    point = np.asarray(point, dtype=float)
    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)

    if (
        point.shape != start.shape
        or point.shape != end.shape
    ):
        raise ValueError(
            "point, start, and end must have the same shape."
        )

    # Segment(s) - point = offset + s * direction.
    offset = start - point
    direction = end - start

    # max_i |offset_i + s * direction_i| can be written as
    # the maximum of the following 2N affine functions.
    slopes = np.concatenate(
        [direction, -direction]
    )

    intercepts = np.concatenate(
        [offset, -offset]
    )

    candidates = [0.0, 1.0]

    # Include all intersections between affine pieces.
    for i in range(len(slopes)):
        for j in range(i + 1, len(slopes)):
            denominator = slopes[i] - slopes[j]

            if abs(denominator) < 1e-15:
                continue

            s = (
                intercepts[j] - intercepts[i]
            ) / denominator

            if 0.0 <= s <= 1.0:
                candidates.append(float(s))

    distances = []

    for s in candidates:
        segment_point = start + s * direction

        distances.append(
            linf_distance(point, segment_point)
        )

    return min(distances)


# Geometry forbidden to a selected junction

# The detector cube is centered inside the unit cube.
def detector_center(N):
    return np.full(N, 0.5)


def cube_radius_to_boundary(center):
    """
    Largest radius r for which the centered L-infinity cube

        {z : ||z - center||_infinity <= r}

    remains inside [0, 1]^N.
    """
    center = np.asarray(center, dtype=float)

    return float(
        np.min(
            np.minimum(center, 1.0 - center)
        )
    )


def forbidden_distance(
    tree,
    junction_name,
    center,
):
    vertices = tree["vertices"]
    edges = tree["edges"]

    distances = []

    for name, position in vertices.items():
        if name != junction_name:
            distances.append(
                linf_distance(center, position)
            )

    for child, parent in edges:
        if junction_name not in {child, parent}:
            distances.append(
                linf_distance_point_to_segment(
                    center,
                    vertices[child],
                    vertices[parent],
                )
            )

    if not distances:
        return np.inf

    return min(distances)

# Detector data

def detector_data(
    tree,
    target_type,
    detector_radius=0.05,
    *,
    center=None,
    tie_tolerance=1e-12,
):
    """
    Find the unique isolatable junction of target_type.

    Returns None for the quotient basepoint, or diagnostic data
    for exactly one admissible junction.
    """
    if target_type not in {"a", "b"}:
        raise ValueError(
            "target_type must be 'a' or 'b'."
        )

    vertices = tree["vertices"]
    vertex_type = tree["vertex_type"]
    edges = tree["edges"]

    if not vertices:
        return None

    N = next(
        iter(vertices.values())
    ).size

    if center is None:
        center = np.full(N, 0.5)
    else:
        center = np.asarray(center, dtype=float)

    ambient_radius = np.min(
        np.minimum(center, 1.0 - center)
    )

    target_names = [
        name
        for name, kind in vertex_type.items()
        if kind == target_type
    ]

    candidates = []

    for name in target_names:
        position = np.asarray(
            vertices[name],
            dtype=float,
        )

        distance_to_center = linf_distance(
            position,
            center,
        )

        forbidden_radius = forbidden_distance(
            tree,
            name,
            center,
        )

        # The detector is allowed to use only a central L-infinity cube
        # of radius detector_radius around `center`.
        if detector_radius <= 0.0:
            raise ValueError(
                "detector_radius must be positive."
            )

        if detector_radius > ambient_radius:
            raise ValueError(
        "detector_radius must not exceed the radius "
        "available inside the ambient cube."
                )

# The outer radius is constrained both by forbidden geometry and
# by the prescribed maximum detector size.
        outer_radius = min(
                ambient_radius,
                forbidden_radius,
                detector_radius,
                )

# A valid detector cube must contain the candidate strictly in
# its interior and exclude every forbidden object.
        if not (
                distance_to_center
                < outer_radius - tie_tolerance
                ):
            continue

        # Choose a cube strictly between the candidate and the nearest
        # obstruction / permitted detector boundary.
        cube_radius = 0.5 * (
                            distance_to_center
                            + outer_radius
                            )

        normalized_position = (
            0.5
            + (
                position - center
            ) / (2.0 * cube_radius)
        )

        normalized_position = np.asarray(
            normalized_position,
            dtype=float,
        )

        # Numerical safety check.
        if np.any(
            normalized_position <= tie_tolerance
        ):
            continue

        if np.any(
            normalized_position >= 1.0 - tie_tolerance
        ):
            continue

        candidates.append({
            "name": name,
            "position": position.copy(),
            "distance_to_center": distance_to_center,
            "forbidden_distance": forbidden_radius,
            "outer_radius": outer_radius,
            "cube_radius": cube_radius,
            "normalized_position": normalized_position,
        })

    if len(candidates) != 1:
        return None

    return candidates[0]

# Now we make the actual maps that detect either an a-junction or a b-junction.
# These maps have values in the quotient of a cube by its boundary.
# These values are either None (the basepoint) or an array which represents an interior point of C^N.

def detect_a(tree, **kwargs):
    """
    Detect a uniquely isolatable a-junction.

    Returns:
        None for the quotient basepoint, or
        y in (0, 1)^N for the detected normalized position.
    """
    data = detector_data(
        tree,
        target_type="a",
        **kwargs,
    )

    if data is None:
        return None

    return data["normalized_position"]


def detect_b(tree, **kwargs):
    """
    Detect a uniquely isolatable b-junction.

    Returns:
        None for the quotient basepoint, or
        y in (0, 1)^N for the detected normalized position.
    """
    data = detector_data(
        tree,
        target_type="b",
        **kwargs,
    )

    if data is None:
        return None

    return data["normalized_position"]
