# Just define our actual map by composing the tree building and detector maps.

import matplotlib.pyplot as plt
import numpy as np
from mathmodels.double_junction.trees.Tree_building import tree_at
from double_junction.detector.Detector_maps import detect_a, detect_b

def make_tree_detector_map(
    family,
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
        raise ValueError(
            "detector_type must be 'a' or 'b'."
        )

    if detector_kwargs is None:
        detector_kwargs = {
            "detector_radius": 0.25,
        }

    N = family["N"]

    if detector_type == "a":
        detector = detect_a
    else:
        detector = detect_b

    def f(p):
        p = np.asarray(p, dtype=float)

        if p.shape != (N + 1,):
            raise ValueError(
                f"p must have shape ({N + 1},), "
                f"got {p.shape}."
            )

        x = p[:N]
        t = float(p[N])

        tree = tree_at(
            family,
            x,
            t,
        )

        return detector(
            tree,
            **detector_kwargs,
        )

    return f
