
"""Here we want to estimate the degree of the detector.
Recall that 
For a cube input p in [0,1]^N, this uses only the two faces p_N=0
and p_N=1.  It assumes all other boundary faces map to the basepoint
in the quotient.  On each contributing face, the detector produces a
map [0,1]^(N-1) / boundary -> itself; its degree is computed using a
regular-value / oriented-Jacobian estimate. There is more info on how this works in the mathcontext folder.
"""

import numpy as np
import matplotlib.pyplot as plt

from dimension_one.detector_one import make_slice_detector_map

##  Distance from q in [0,1]^d to its boundary.

def distance_to_cube_boundary(q):
    q = np.asarray(q, dtype=float)
    return float(np.min(np.minimum(q, 1.0 - q)))


"""Return the output if it is strictly inside its transverse cube.

    ``None`` is used for the quotient basepoint.  Any output on the
    transverse boundary also represents the quotient basepoint.
"""

def detector_output_interior(detector_map, p, *, boundary_tol=1e-9):
    q = detector_map(np.asarray(p, dtype=float))
    if q is None:
        return None

    q = np.asarray(q, dtype=float)
    if not np.all(np.isfinite(q)):
        return None
    if np.any(q <= boundary_tol) or np.any(q >= 1.0 - boundary_tol):
        return None
    return q

# We restrict our detector map to certain faces.

def _face_map(detector_map, d, time_value, x, boundary_tol):
    p = np.empty(d + 1, dtype=float)
    p[:d] = x
    p[d] = time_value
    return detector_output_interior(
        detector_map,
        p,
        boundary_tol=boundary_tol,
    )

# We compute the associated Jacobians

def _jacobian_on_face(face_map, x, h, d):
    """Central-difference Jacobian, or None if we hit basepoint."""
    jac = np.empty((d, d), dtype=float)

    for j in range(d):
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

"""Now we can estimate the relative degree on both time faces.

    Searches a uniform grid for cells whose detector outputs surround
    ``regular_value``.  A Newton refinement in the cell gives a
    preimage; each preimage contributes sign(det(Df)).

    May encounter problems with the selection of the regular value - 
    although our map here is nice, this will be more of a problem for the complicated junctions.
"""

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
   
    if N < 2:
        raise ValueError("N must be at least 2.")
    if time_value not in (0.0, 1.0):
        raise ValueError("time_value must be 0.0 or 1.0.")
    if samples < 3:
        raise ValueError("samples must be at least 3.")

    d = N - 1
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

    def face_map(x):
        return _face_map(
            detector_map,
            d,
            time_value,
            x,
            boundary_tol,
        )

    # Candidate cells use an output bounding-box test.  This is an
    # estimator, not a certified simplicial degree computation.
    grid = np.linspace(0.0, 1.0, samples)
    cache = {}

    def cached(x):
        key = tuple(np.round(x, 14))
        if key not in cache:
            cache[key] = face_map(x)
        return cache[key]

    roots = []

    for index in np.ndindex(*(samples - 1 for _ in range(d))):
        lo = np.array(index, dtype=float) * h_grid
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
        if np.any(regular_value < corner_values.min(axis=0)):
            continue
        if np.any(regular_value > corner_values.max(axis=0)):
            continue

        # Start at cell center and use constrained Newton iterations.
        x = lo + 0.5 * h_grid
        for _ in range(20):
            q = face_map(x)
            jac = _jacobian_on_face(face_map, x, h_diff, d)
            if q is None or jac is None:
                break
            try:
                step = np.linalg.solve(jac, q - regular_value)
            except np.linalg.LinAlgError:
                break
            x_next = x - step
            if np.any(x_next < lo - 1e-10) or np.any(x_next > lo + h_grid + 1e-10):
                break
            if np.linalg.norm(x_next - x) < 1e-11:
                x = x_next
                break
            x = x_next

        q = face_map(x)
        jac = _jacobian_on_face(face_map, x, h_diff, d)
        if q is None or jac is None:
            continue
        if np.linalg.norm(q - regular_value) > root_tol:
            continue

        # Avoid duplicate roots from neighbouring candidate cells.
        if any(np.linalg.norm(x - old) < 0.25 * h_grid for old in roots):
            continue
        roots.append(x)

    signs = []
    determinants = []
    for x in roots:
        jac = _jacobian_on_face(face_map, x, h_diff, d)
        determinant = float(np.linalg.det(jac))
        determinants.append(determinant)
        signs.append(int(np.sign(determinant)) if abs(determinant) > 1e-8 else 0)

    return {
        "time_face": time_value,
        "relative_degree": int(sum(signs)),
        "preimages": np.asarray(roots),
        "jacobian_determinants": np.asarray(determinants),
        "local_degrees": np.asarray(signs, dtype=int),
        "regular_value": regular_value,
        "samples": samples,
    }


"""Now we put all of this together to estimate the degree of the boundary detector map.

    Assumption: the side faces p_i=0 or 1 for i < N are sent to the
    quotient basepoint, so only p_N=0 and p_N=1 contribute.

    With the standard boundary orientation of [0,1]^N, the t=1 face
    has positive orientation and the t=0 face has negative orientation, so degree = degree(t=1) - degree(t=0).
"""

def detector_boundary_degree(
    family,
    *,
    regular_value=None,
    samples=61,
    boundary_tol=1e-7,
):

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
