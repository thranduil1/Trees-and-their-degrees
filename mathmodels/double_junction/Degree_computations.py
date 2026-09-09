# Now we implement a code to compute the degree of the detector map.
# Recall that the detector map is a map from the boundary of I^{N+1} to I^N quotiented by its boundary.
# Its domain is the faces of I^{N+1} of which there are 2N+2.
# However, we notice that the detector map is just the constant map on the basepoint on 2N of these faces. 
# Therefore, it suffices to compute the degree on the two remaining faces.
# (This is explained in some more detail in the mathcontext folder).

import numpy as np
from double_junction.make_detector_map import make_tree_detector_map

# The first thing we do is restrict the detector to the only two face maps that we care about.
def _face_map_from_tree_detector(
    tree_detector_map,
    N,
    time_value,
    x,
    tie_tolerance=1e-12,
):
    """Evaluate the detector on a fixed time face.

    Parameters
    ----------
    tree_detector_map : callable
        f(p) where p = (x, t) in [0,1]^{N+1}.
    N : int
        Transverse dimension (codomain cube dimension).
    time_value : {0.0, 1.0}
        Which time face to evaluate.
    x : ndarray
        Point in [0,1]^N (the spatial coordinates).
    tie_tolerance : float
        Passed through to the detector if needed.

    Returns
    -------
    q : ndarray or None
        Point in (0,1)^N or None (basepoint).
    """
    p = np.empty(N + 1, dtype=float)
    p[:N] = x
    p[N] = time_value

    q = tree_detector_map(p)

    if q is None:
        return None

    q = np.asarray(q, dtype=float)

    # Treat points extremely close to the boundary as basepoint.
    if np.any(q <= tie_tolerance) or np.any(q >= 1.0 - tie_tolerance):
        return None

    return q

# We will need to compute the Jacobian

def _jacobian_on_face(face_map, x, h, N):
    """Central-difference Jacobian on the time face.

    Returns None if any stencil point hits the basepoint.
    """
    jac = np.empty((N, N), dtype=float)

    for j in range(N):
        xp = x.copy()
        xm = x.copy()
        xp[j] += h
        xm[j] -= h

        fp = face_map(xp)
        fm = face_map(xm)

        if fp is None or fm is None:
            return None

        jac[:, j] = (fp - fm) / (2.0 * h)

    return jac

# Computes the degree of the map on each time face.

def degree_on_time_face(
    tree_detector_map,
    *,
    N,
    time_value,
    regular_value=None,
    samples=61,
    tie_tolerance=1e-12,
    root_tol=None,
):
    """Estimate the relative degree on the face t = time_value.

    This uses a grid search for cells whose images surround
    ``regular_value``, then refines preimages with Newton iterations.

    Returns a dict with:
        - "time_face": time_value
        - "relative_degree": integer
        - "preimages": array of x-positions in [0,1]^N
        - "jacobian_determinants", "local_degrees"
        - "regular_value"
        - "samples"
    """
    if N < 3:
        raise ValueError("N must be at least 3.")
    if time_value not in (0.0, 1.0):
        raise ValueError("time_value must be 0.0 or 1.0.")
    if samples < 3:
        raise ValueError("samples must be at least 3.")

    if regular_value is None:
        regular_value = np.full(N, 0.5)
    regular_value = np.asarray(regular_value, dtype=float)

    if regular_value.shape != (N,):
        raise ValueError(
            f"regular_value must have shape ({N},)."
        )
    if np.any(regular_value <= 0.0) or np.any(regular_value >= 1.0):
        raise ValueError(
            "regular_value must lie in the open cube."
        )

    h_grid = 1.0 / (samples - 1)
    h_diff = min(0.25 * h_grid, 1e-4)
    if root_tol is None:
        root_tol = 0.05 * h_grid

    def face_map(x):
        return _face_map_from_tree_detector(
            tree_detector_map,
            N,
            time_value,
            x,
            tie_tolerance=tie_tolerance,
        )

    grid = np.linspace(0.0, 1.0, samples)
    cache = {}

    def cached(x):
        key = tuple(np.round(x, 14))
        if key not in cache:
            cache[key] = face_map(x)
        return cache[key]

    roots = []

    # Scan all grid cells.
    for index in np.ndindex(*(samples - 1 for _ in range(N))):
        lo = np.array(index, dtype=float) * h_grid

        # Evaluate at all 2^N corners.
        corners = []
        for bits in np.ndindex(*(2 for _ in range(N))):
            x = lo + h_grid * np.array(bits, dtype=float)
            q = cached(x)
            if q is None:
                corners = None
                break
            corners.append(q)

        if corners is None:
            continue

        corner_values = np.asarray(corners)

        # Quick bounding-box test: does the image cell contain regular_value?
        if np.any(regular_value < corner_values.min(axis=0)):
            continue
        if np.any(regular_value > corner_values.max(axis=0)):
            continue

        # Newton refinement from the cell center.
        x = lo + 0.5 * h_grid

        for _ in range(20):
            q = face_map(x)
            jac = _jacobian_on_face(face_map, x, h_diff, N)
            if q is None or jac is None:
                break
            try:
                step = np.linalg.solve(jac, q - regular_value)
            except np.linalg.LinAlgError:
                break
            x_next = x - step

            # Stay inside the cell.
            if np.any(x_next < lo - 1e-10) or np.any(x_next > lo + h_grid + 1e-10):
                break

            if np.linalg.norm(x_next - x) < 1e-11:
                x = x_next
                break

            x = x_next

        q = face_map(x)
        jac = _jacobian_on_face(face_map, x, h_diff, N)
        if q is None or jac is None:
            continue
        if np.linalg.norm(q - regular_value) > root_tol:
            continue

        # Avoid duplicate roots from neighboring candidate cells.
        if any(np.linalg.norm(x - old) < 0.25 * h_grid for old in roots):
            continue
        roots.append(x)

    signs = []
    determinants = []

    for x in roots:
        jac = _jacobian_on_face(face_map, x, h_diff, N)
        determinant = float(np.linalg.det(jac))
        determinants.append(determinant)
        signs.append(
            int(np.sign(determinant))
            if abs(determinant) > 1e-8
            else 0
        )

    return {
        "time_face": time_value,
        "relative_degree": int(sum(signs)),
        "preimages": np.asarray(roots),
        "jacobian_determinants": np.asarray(determinants),
        "local_degrees": np.asarray(signs, dtype=int),
        "regular_value": regular_value,
        "samples": samples,
    }

# The total degree is obtained by degree(t=1)-degree(t=0).
# We are using different regular values to compute the degree on each face.
# Usually, using different regular values on the two faces does NOT
# directly give a topological degree. This is only valid if you
# can justify that both regular values lie in the same connected
# component of regular values and the map is proper, but here this is the case !

def tree_detector_boundary_degree(
    family,
    detector_type,
    *,
    regular_value_bottom=None,
    regular_value_top=None,
    samples=61,
    tie_tolerance=1e-12,
    detector_kwargs=None,
):
    """Estimate the degree with separate regular values for t=0 and t=1.

    This is useful when the detector outputs on the two faces are
    concentrated in different regions of the cube.

    Parameters
    ----------
    family : dict
        From make_tree_family.
    detector_type : {"a", "b"}
        Which detector to use.
    regular_value_bottom : ndarray or None
        Regular value for t=0 face. Defaults to center.
    regular_value_top : ndarray or None
        Regular value for t=1 face. Defaults to center.
    samples : int
        Grid resolution per face.
    tie_tolerance : float
        Tolerance for treating points as basepoint.
    detector_kwargs : dict or None
        Extra kwargs for detector_data.

    Returns
    -------
    result : dict
        Same structure as tree_detector_boundary_degree.
    """
    

    if detector_type not in {"a", "b"}:
        raise ValueError(
            "detector_type must be 'a' or 'b'."
        )

    N = int(family["N"])

    if N < 3:
        raise ValueError("N must be at least 3.")

    tree_detector_map = make_tree_detector_map(
        family,
        detector_type,
       detector_kwargs={
        "detector_radius": 0.25,
    },
    )

    bottom = degree_on_time_face(
        tree_detector_map,
        N=N,
        time_value=0.0,
        regular_value=regular_value_bottom,
        samples=samples,
        tie_tolerance=tie_tolerance,
    )

    top = degree_on_time_face(
        tree_detector_map,
        N=N,
        time_value=1.0,
        regular_value=regular_value_top,
        samples=samples,
        tie_tolerance=tie_tolerance,
    )

    degree = top["relative_degree"] - bottom["relative_degree"]

    return {
        "degree": int(degree),
        "top_face": top,
        "bottom_face": bottom,
        "orientation_convention": "degree = degree(t=1) - degree(t=0)",
        "regular_value_bottom": bottom["regular_value"],
        "regular_value_top": top["regular_value"],
    }

