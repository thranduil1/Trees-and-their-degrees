import numpy as np


def _jacobian_central_diff(g, x, h, d):
    """Central-difference Jacobian of g at x in R^d.

    Returns None if any stencil point hits the basepoint.
    """
    jac = np.empty((d, d), dtype=float)

    for j in range(d):
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
    d,
    regular_value=None,
    samples=61,
    boundary_tol=1e-7,
    root_tol=None,
):
    """Estimate the relative degree of a map g:[0,1]^d -> [0,1]^d.

    The map g may return None to indicate the quotient basepoint.
    Any output with a coordinate <= boundary_tol or >= 1 - boundary_tol
    is also treated as basepoint.

    Parameters
    ----------
    g : callable
        Function x -> q or None, with x,q in [0,1]^d.
    d : int
        Dimension of domain and codomain.
    regular_value : ndarray or None
        Target regular value in (0,1)^d. Defaults to center.
    samples : int
        Number of grid points in each direction.
    boundary_tol : float
        Tolerance for treating outputs as basepoint.
    root_tol : float or None
        Tolerance for accepting a refined preimage.

    Returns
    -------
    result : dict
        {
            "relative_degree": int,
            "preimages": ndarray of shape (k, d),
            "jacobian_determinants": ndarray,
            "local_degrees": ndarray of int,
            "regular_value": ndarray,
            "samples": int,
        }
    """
    if d < 1:
        raise ValueError("d must be at least 1.")
    if samples < 3:
        raise ValueError("samples must be at least 3.")

    if regular_value is None:
        regular_value = np.full(d, 0.5)
    regular_value = np.asarray(regular_value, dtype=float)

    if regular_value.shape != (d,):
        raise ValueError(f"regular_value must have shape ({d},).")
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
        if not np.all(np.isfinite(q)):
            return None
        if np.any(q <= boundary_tol) or np.any(q >= 1.0 - boundary_tol):
            return None
        return q

    cache = {}

    def cached(x):
        key = tuple(np.round(x, 14))
        if key not in cache:
            cache[key] = safe_g(x)
        return cache[key]

    roots = []

    # Scan all grid cells.
    for index in np.ndindex(*(samples - 1 for _ in range(d))):
        lo = np.array(index, dtype=float) * h_grid

        # Evaluate at all 2^d corners.
        corners = []
        for bits in np.ndindex(*(2 for _ in range(d))):
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
            jac = _jacobian_central_diff(safe_g, x, h_diff, d)
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
        jac = _jacobian_central_diff(safe_g, x, h_diff, d)
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
        jac = _jacobian_central_diff(safe_g, x, h_diff, d)
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
    detector_map,
    *,
    N,
    time_value,
    regular_value=None,
    samples=61,
    boundary_tol=1e-7,
    root_tol=None,
):
    """Estimate the relative degree on the face p_N = time_value.

    Parameters
    ----------
    detector_map : callable
        Map p -> q or None, with p in [0,1]^N, q in [0,1]^{N-1}.
    N : int
        Full input dimension (spatial + time).
    time_value : {0.0, 1.0}
        Which time face to evaluate.
    regular_value, samples, boundary_tol, root_tol:
        Passed to degree_on_cube_map.

    Returns
    -------
    result : dict
        Same structure as degree_on_cube_map, plus "time_face".
    """
    if N < 2:
        raise ValueError("N must be at least 2.")
    if time_value not in (0.0, 1.0):
        raise ValueError("time_value must be 0.0 or 1.0.")

    d = N - 1

    def face_map(x):
        p = np.empty(N, dtype=float)
        p[:d] = x
        p[d] = time_value
        q = detector_map(np.asarray(p, dtype=float))
        return q  # None is allowed; degree_on_cube_map will handle it

    result = degree_on_cube_map(
        face_map,
        d=d,
        regular_value=regular_value,
        samples=samples,
        boundary_tol=boundary_tol,
        root_tol=root_tol,
    )

    return {
        "time_face": time_value,
        **result,
    }

from simple_junction.detector_simple import make_slice_detector_map


def detector_boundary_degree(
    family,
    *,
    regular_value=None,
    samples=61,
    boundary_tol=1e-7,
):
    """Compute the boundary degree for the simple-junction detector.

    Assumption: side faces p_i=0 or 1 for i < N map to the quotient
    basepoint, so only p_N=0 and p_N=1 contribute.

    With the standard boundary orientation of [0,1]^N, the p_N=1 face
    has positive orientation and p_N=0 has negative orientation, so
    degree = degree(p_N=1) - degree(p_N=0).
    """
    N = int(family["N"])
    detector_map = make_slice_detector_map(family)

    bottom = degree_on_time_face(
        detector_map,
        N=N,
        time_value=0.0,
        regular_value=regular_value,
        samples=samples,
        boundary_tol=boundary_tol,
    )

    top = degree_on_time_face(
        detector_map,
        N=N,
        time_value=1.0,
        regular_value=regular_value,
        samples=samples,
        boundary_tol=boundary_tol,
    )

    degree = top["relative_degree"] - bottom["relative_degree"]

    return {
        "degree": int(degree),
        "top_face": top,
        "bottom_face": bottom,
        "orientation_convention": "degree = degree(p_N=1) - degree(p_N=0)",
    }