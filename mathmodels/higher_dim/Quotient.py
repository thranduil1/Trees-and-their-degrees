"""
Here we simply make explicit the conversion between a quotient-cube point 
to a point of S^N in R^(N+1).

    Input
    -----
    q:
        None            = quotient basepoint
        array in (0,1)^N = ordinary quotient representative

    Output
    ------
    Unit vector in R^(N+1).

    Convention
    ----------
    The quotient basepoint is sent to the north pole:

        (0, ..., 0, 1).

Every point of boundary(C^N) would also go to this north pole.
"""
# This will be useful to plot the detector (where it doesn't matter much)
# and to compute the degree (where it matters a lot).
import numpy as np
import matplotlib.pyplot as plt

def quotient_cube_to_sphere(q, N):

    north_pole = np.zeros(N + 1)
    north_pole[-1] = 1.0

    if q is None:
        return north_pole

    q = np.asarray(q, dtype=float)

    if q.shape != (N,):
        raise ValueError(
            f"q must have shape ({N},), got {q.shape}."
        )

    if np.any(q < 0.0) or np.any(q > 1.0):
        raise ValueError(
            "q must lie in [0,1]^N."
        )

    # First move [0,1]^N to [-1,1]^N.
    u = 2.0 * q - 1.0

    r_infinity = float(np.max(np.abs(u)))

    # The center of the cube maps to the south pole.
    if r_infinity < 1e-14:
        south_pole = np.zeros(N + 1)
        south_pole[-1] = -1.0
        return south_pole

    # Radially identify the cube with the Euclidean unit ball.
    #
    # If u lies on boundary([-1,1]^N), then r_infinity = 1,
    # so v lies on the Euclidean unit sphere.
    u_norm = float(np.linalg.norm(u))

    v = r_infinity * u / u_norm

    rho = float(np.linalg.norm(v))
    rho = min(max(rho, 0.0), 1.0)

    # Collapse the boundary of the unit ball to the north pole.
    #
    # rho = 0 -> south pole
    # rho = 1 -> north pole
    horizontal = (
        2.0
        * np.sqrt(max(0.0, 1.0 - rho * rho))
        * v
    )

    vertical = 2.0 * rho * rho - 1.0

    sphere_point = np.concatenate(
        [horizontal, [vertical]]
    )

    # Remove negligible numerical drift.
    return sphere_point / np.linalg.norm(sphere_point)

