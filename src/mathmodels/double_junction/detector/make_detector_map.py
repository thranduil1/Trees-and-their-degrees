# Just define our actual map by composing the tree building and detector maps.

import numpy as np

from mathmodels.double_junction.detector.detector_maps import (
    detect_a,
    detect_b,
)
from mathmodels.double_junction.trees.tree_building import make_double_geometric_tree_at


def make_tree_detector_map(
    double_combinatorial_tree,
    detector_type,
    *,
    detector_kwargs=None,
):
    """
    Construct the composite map

        f : C^(N+1) -> C^N / boundary(C^N)

    defined by:

        p = (x, t)
        (x, t) -> embedded tree
        embedded tree -> detector output.

    Returns:
        None            for the quotient basepoint
        NumPy array     for a point in (0, 1)^N
    """

    if detector_type not in {"a", "b"}:
        raise ValueError("detector_type must be 'a' or 'b'.")

    if detector_kwargs is None:
        detector_kwargs = {
            "detector_radius": 0.05,
        }

    N = double_combinatorial_tree.N

    if detector_type == "a":
        detector = detect_a
    else:
        detector = detect_b

    def f(p):
        p = np.asarray(p, dtype=float)

        if p.shape != (N + 1,):
            raise ValueError(f"p must have shape ({N + 1},), got {p.shape}.")

        x = p[:N]
        t = float(p[N])

        double_geometric_tree = make_double_geometric_tree_at(
            double_combinatorial_tree,
            x,
            t,
        )

        return detector(
            double_geometric_tree,
            **detector_kwargs,
        )

    return f
