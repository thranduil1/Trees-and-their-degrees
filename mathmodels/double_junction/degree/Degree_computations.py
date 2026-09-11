# Now we implement a code to compute the degree of the detector map.
# Recall that the detector map is a map from the boundary of I^{N+1} to I^N quotiented by its boundary.
# Its domain is the faces of I^{N+1} of which there are 2N+2.
# However, we notice that the detector map is just the constant map on the basepoint on 2N of these faces. 
# Therefore, it suffices to compute the degree on the two remaining faces.
# (This is explained in some more detail in the mathcontext folder).

import numpy as np

def _jacobian_central_diff(g, x, h, N):
    """Central-difference Jacobian of g at x.

    Returns None if any stencil point hits the basepoint.
    """
    jac = np.empty((N, N), dtype=float)

    for j in range(N):
        xp = x.copy()
        xm = x.copy()
        xp[j] += h
        xm[j] -= h

        fp = g(xp)
        fm = g(xm)

        if fp is None or fm is None:
            return None

        jac[:, j] = (fp - fm) / (2.0 * h)

    return jac

def degree_on_cube_map(
    g,
    *,
    N,
    regular_value=None,
    samples=61,
    tie_tolerance=1e-12,
    root_tol=None,
):
    """Estimate the relative degree of a map g:[0,1]^N -> [0,1]^N.

    The map g is allowed to return None to indicate the basepoint
    (i.e. points on the quotient boundary). Those are excluded from
    the degree computation.

    Parameters
    ----------
    g : callable
        Function x -> q or None, where x,q are in [0,1]^N.
    N : int
        Dimension of the domain and codomain cube.
    regular_value : ndarray or None
        Target regular value in (0,1)^N. Defaults to center.
    samples : int
        Number of grid points in each direction.
    tie_tolerance : float
        If g(x) has any coordinate <= tie_tolerance or
        >= 1 - tie_tolerance, it is treated as basepoint (None).
    root_tol : float or None
        Tolerance for accepting a refined preimage.

    Returns
    -------
    result : dict
        {
            "relative_degree": int,
            "preimages": ndarray of shape (k, N),
            "jacobian_determinants": ndarray,
            "local_degrees": ndarray of int,
            "regular_value": ndarray,
            "samples": int,
        }
    """
    if N < 1:
        raise ValueError("N must be at least 1.")
    if samples < 3:
        raise ValueError("samples must be at least 3.")

    if regular_value is None:
        regular_value = np.full(N, 0.5)
    regular_value = np.asarray(regular_value, dtype=float)

    if regular_value.shape != (N,):
        raise ValueError(f"regular_value must have shape ({N},).")
    if np.any(regular_value <= 0.0) or np.any(regular_value >= 1.0):
        raise ValueError("regular_value must lie in the open cube.")

    h_grid = 1.0 / (samples - 1)
    h_diff = min(0.25 * h_grid, 1e-4)
    if root_tol is None:
        root_tol = 0.05 * h_grid

    def safe_g(x):
        q = g(x)
        if q is None:
            return None
        q = np.asarray(q, dtype=float)
        if np.any(q <= tie_tolerance) or np.any(q >= 1.0 - tie_tolerance):
            return None
        return q

    grid = np.linspace(0.0, 1.0, samples)
    cache = {}

    def cached(x):
        key = tuple(np.round(x, 14))
        if key not in cache:
            cache[key] = safe_g(x)
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

        # Quick bounding-box test.
        if np.any(regular_value < corner_values.min(axis=0)):
            continue
        if np.any(regular_value > corner_values.max(axis=0)):
            continue

        # Newton refinement from the cell center.
        x = lo + 0.5 * h_grid

        for _ in range(20):
            q = safe_g(x)
            jac = _jacobian_central_diff(safe_g, x, h_diff, N)
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

        q = safe_g(x)
        jac = _jacobian_central_diff(safe_g, x, h_diff, N)
        if q is None or jac is None:
            continue
        if np.linalg.norm(q - regular_value) > root_tol:
            continue

        # Avoid duplicate roots from neighboring cells.
        if any(np.linalg.norm(x - old) < 0.25 * h_grid for old in roots):
            continue
        roots.append(x)

    signs = []
    determinants = []

    for x in roots:
        jac = _jacobian_central_diff(safe_g, x, h_diff, N)
        determinant = float(np.linalg.det(jac))
        determinants.append(determinant)
        signs.append(
            int(np.sign(determinant))
            if abs(determinant) > 1e-8
            else 0
        )

    return {
        "relative_degree": int(sum(signs)),
        "preimages": np.asarray(roots),
        "jacobian_determinants": np.asarray(determinants),
        "local_degrees": np.asarray(signs, dtype=int),
        "regular_value": regular_value,
        "samples": samples,
    }

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

    Parameters
    ----------
    tree_detector_map : callable
        f(p) where p = (x, t) in [0,1]^{N+1}.
    N : int
        Transverse dimension (codomain cube dimension).
    time_value : {0.0, 1.0}
        Which time face to evaluate.
    regular_value, samples, tie_tolerance, root_tol:
        Passed to degree_on_cube_map.

    Returns
    -------
    result : dict
        Same structure as degree_on_cube_map, plus "time_face".
    """
    if time_value not in (0.0, 1.0):
        raise ValueError("time_value must be 0.0 or 1.0.")

    def face_map(x):
        p = np.empty(N + 1, dtype=float)
        p[:N] = x
        p[N] = time_value
        q = tree_detector_map(p)
        return q  # None is allowed; degree_on_cube_map will wrap it

    result = degree_on_cube_map(
        face_map,
        N=N,
        regular_value=regular_value,
        samples=samples,
        tie_tolerance=tie_tolerance,
        root_tol=root_tol,
    )

    return {
        "time_face": time_value,
        **result,
    }

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
    """Estimate the degree with separate regular values for t=0 and t=1."""

    if detector_type not in {"a", "b"}:
        raise ValueError("detector_type must be 'a' or 'b'.")

    N = int(family["N"])
    if N < 3:
        raise ValueError("N must be at least 3.")

    if detector_kwargs is None:
        detector_kwargs = {
            "detector_radius": 0.25,
        }

    from double_junction.detector.make_detector_map import make_tree_detector_map

    tree_detector_map = make_tree_detector_map(
        family,
        detector_type,
        detector_kwargs=detector_kwargs,
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