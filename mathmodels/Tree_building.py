## Building a model for our embeddings of trees into cubes ##

import numpy as np

# Basic geometry stuff

# First: a smooth interpolation [0, 1] -> [0, 1].

def smoothstep(s):
    return s * s * (3.0 - 2.0 * s)

# A function which returns equally spaced points in an interval [margin, 1 - margin]

def equally_spaced_interior(count, margin):
    if count < 1:
        raise ValueError("count must be positive.")

    if count == 1:
        return np.array([0.5])

    return np.linspace(
        margin,
        1.0 - margin,
        count,
    )

# Make a tree family depending on some parameters:
"""
    a, b:
        The leaves form an a-by-b rectangular array.

    N:
        Ambient cube dimension.
        coordinate 0     = a-direction,
        coordinate 1     = b-direction,
        coordinate N - 1 = time direction.

    barycenter_margin:
        The physical junction barycenter lies in

            [margin, 1-margin]^N.

        This leaves room around junctions for their spread and,
        later, for detector cubes.

    junction_span_a, junction_span_b:
        Total spans of the lower-junction arrays in the a- and
        b-directions.  
        
    leaf_time, root_time:
        Fixed time coordinates of leaves and root.
    """

def make_tree_family(
    a,
    b,
    *,
    N=3,
    barycenter_margin=0.1,
    junction_span_a=0.20,
    junction_span_b=0.20,
    junction_time_gap=0.1,
    leaf_margin=0.10,
    leaf_time=0.05,
    root_time=0.95,
):
    
    if a < 1 or b < 1:
        raise ValueError("a and b must be positive.")

    if N < 3:
        raise ValueError("N must be at least 3.")

    if not 0.0 < barycenter_margin < 0.5:
        raise ValueError(
            "barycenter_margin must lie in (0, 1/2)."
        )

    if not 0.0 <= leaf_margin < 0.5:
        raise ValueError(
            "leaf_margin must lie in [0, 1/2)."
        )

    if not 0.0 <= leaf_time < barycenter_margin:
        raise ValueError(
            "leaf_time must be below barycenter_margin."
        )

    if not 1.0 - barycenter_margin < root_time <= 1.0:
        raise ValueError(
            "root_time must lie above 1 - barycenter_margin."
        )
# Below are some extra tests which make this definition more accurate but also more annoying to work this so we remove them.
  #  if not 0.0 <= junction_span_a < 2.0 * barycenter_margin:
  #      raise ValueError(
    #         "junction_span_a is too large for the interior margin."
   #     )

  #  if not 0.0 <= junction_span_b < 2.0 * barycenter_margin:
   #     raise ValueError(
   #         "junction_span_b is too large for the interior margin."
   #     )
        
    # Adding a check that there's a time gap
        
  #  if not 0.0 < junction_time_gap < barycenter_margin:
  #      raise ValueError(
  #          "junction_time_gap must lie in "
  #          "(0, barycenter_margin)."
  #      )

    max_upper_offset = max(
        a * junction_time_gap / (a + 1),
        b * junction_time_gap / (b + 1),
    )

    max_lower_offset = max(
        junction_time_gap / (a + 1),
        junction_time_gap / (b + 1),
    )
# these are again tests that make sense but that I choose not to care about right now
  #  if leaf_time >= barycenter_margin - max_lower_offset:
   #     raise ValueError(
   #         "leaf_time is too high: leaves must lie below "
   #         "every possible lower junction."
   #     )

  #  if root_time <= 1.0 - barycenter_margin + max_upper_offset:
   #     raise ValueError(
   #         "root_time is too low: root must lie above "
   #         "every possible upper junction."
   #     )

    e_a = np.zeros(N)
    e_a[0] = 1.0

    e_b = np.zeros(N)
    e_b[1] = 1.0

    e_time = np.zeros(N)
    e_time[-1] = 1.0

    leaf_a_coordinates = equally_spaced_interior(
        a,
        leaf_margin,
    )

    leaf_b_coordinates = equally_spaced_interior(
        b,
        leaf_margin,
    )

    leaves = {}

    for i in range(a):
        for j in range(b):
            position = np.full(N, 0.5)

            position[0] = leaf_a_coordinates[i]
            position[1] = leaf_b_coordinates[j]
            position[-1] = leaf_time

            leaves[f"L_{i}_{j}"] = position

    root = np.full(N, 0.5)
    root[-1] = root_time

    return {
        "a": int(a),
        "b": int(b),
        "N": int(N),
        "barycenter_margin": float(barycenter_margin),
        "junction_span_a": float(junction_span_a),
        "junction_span_b": float(junction_span_b),
        "junction_time_gap": float(junction_time_gap),
        "e_a": e_a,
        "e_b": e_b,
        "e_time": e_time,
        "leaves": leaves,
        "root": root,
    }


# Physical barycenter (because of margin complications (x_bar always lies in the interior box
# [margin, 1-margin]^N.)- in reality I actually bypass a lot of this)

def physical_barycenter(family, x):
    N = family["N"]
    margin = family["barycenter_margin"]

    x = np.asarray(x, dtype=float)

    if x.shape != (N,):
        raise ValueError(
            f"x must have shape ({N},), got {x.shape}."
        )

    if np.any(x < 0.0) or np.any(x > 1.0):
        raise ValueError("x must lie in [0, 1]^N.")

    return margin + (1.0 - 2.0 * margin) * x

# Now we construct the map which sends (x, t) to an affine embedded rooted tree. 
# The barycenter is x (essentially) and t governs the type of junction we are looking at.
# No two distinct junctions occupy the same position when t != 1/2. At t = 1/2 there is only one junction: AB.

def tree_at(family, x, t):

    if not 0.0 <= t <= 1.0:
        raise ValueError("t must lie in [0, 1].")

    a = family["a"]
    b = family["b"]

    e_a = family["e_a"]
    e_b = family["e_b"]
    e_time = family["e_time"]

    span_a = family["junction_span_a"]
    span_b = family["junction_span_b"]
    time_gap = family["junction_time_gap"]

    x_bar = physical_barycenter(family, x)

    vertices = {
        name: position.copy()
        for name, position in family["leaves"].items()
    }

    vertices["R"] = family["root"].copy()

    vertex_type = {
        name: "leaf"
        for name in family["leaves"]
    }

    vertex_type["R"] = "root"

    edges = []

    # --------------------------------------------------------
    # t < 1/2:
    #
    # b lower a-junctions A_j, then one upper b-junction B.
    # --------------------------------------------------------
    if t < 0.5:
        collapse = 1.0 - smoothstep(2.0 * t)

        lower_time_offset = (
            -time_gap / (b + 1)
        )

        upper_time_offset = (
            b * time_gap / (b + 1)
        )

        for j in range(b):
            if b == 1:
                spatial_offset = 0.0
            else:
                spatial_offset = (
                    span_b
                    * (
                        j / (b - 1)
                        - 0.5
                    )
                )

            vertices[f"A_{j}"] = (
                x_bar
                + collapse
                * (
                    spatial_offset * e_b
                    + lower_time_offset * e_time
                )
            )

            vertex_type[f"A_{j}"] = "a"

        vertices["B"] = (
            x_bar
            + collapse
            * upper_time_offset
            * e_time
        )

        vertex_type["B"] = "b"

        for i in range(a):
            for j in range(b):
                edges.append(
                    (f"L_{i}_{j}", f"A_{j}")
                )

        for j in range(b):
            edges.append(
                (f"A_{j}", "B")
            )

        edges.append(("B", "R"))

    # --------------------------------------------------------
    # t = 1/2:
    #
    # Exactly one ab-junction.
    # --------------------------------------------------------
    elif t == 0.5:
        vertices["AB"] = x_bar.copy()
        vertex_type["AB"] = "ab"

        for i in range(a):
            for j in range(b):
                edges.append(
                    (f"L_{i}_{j}", "AB")
                )

        edges.append(("AB", "R"))

    # --------------------------------------------------------
    # t > 1/2:
    #
    # a lower b-junctions B_i, then one upper a-junction A.
    # --------------------------------------------------------
    else:
        collapse = 1.0 - smoothstep(
            2.0 * (1.0 - t)
        )

        lower_time_offset = (
            -time_gap / (a + 1)
        )

        upper_time_offset = (
            a * time_gap / (a + 1)
        )

        for i in range(a):
            if a == 1:
                spatial_offset = 0.0
            else:
                spatial_offset = (
                    span_a
                    * (
                        i / (a - 1)
                        - 0.5
                    )
                )

            vertices[f"B_{i}"] = (
                x_bar
                + collapse
                * (
                    spatial_offset * e_a
                    + lower_time_offset * e_time
                )
            )

            vertex_type[f"B_{i}"] = "b"

        vertices["A"] = (
            x_bar
            + collapse
            * upper_time_offset
            * e_time
        )

        vertex_type["A"] = "a"

        for i in range(a):
            for j in range(b):
                edges.append(
                    (f"L_{i}_{j}", f"B_{i}")
                )

        for i in range(a):
            edges.append(
                (f"B_{i}", "A")
            )

        edges.append(("A", "R"))

    return {
        "vertices": vertices,
        "edges": edges,
        "vertex_type": vertex_type,
        "barycenter": x_bar,
    }


# Extra maps to access the junctions and edges 

def junctions(tree):
  
    return [
        (
            name,
            tree["vertices"][name],
            tree["vertex_type"][name],
        )
        for name in tree["vertices"]
        if tree["vertex_type"][name] in {"a", "b", "ab"}
    ]


def edge_segments(tree):
    return [
        (
            child,
            parent,
            tree["vertices"][child],
            tree["vertices"][parent],
        )
        for child, parent in tree["edges"]
    ]
