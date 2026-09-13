## Building a model for our embeddings of trees into cubes ##

import numpy as np

class DoubleCombinatorialTree:
    def __init__(
        self,
        *,
        a,
        b,
        N,
        barycenter_box_margin,
        junction_span_a,
        junction_span_b,
        junction_time_gap,
        e_a,
        e_b,
        e_time,
        leaves,
        root,
    ):
        self.a = a
        self.b=b
        self.N = N
        self.barycenter_box_margin = barycenter_box_margin
        self.junction_span_a = junction_span_a
        self.junction_span_b = junction_span_b
        self.junction_time_gap = junction_time_gap
        self.e_a = e_a
        self.e_b = e_b
        self.e_time = e_time
        self.leaves = leaves
        self.root = root

class DoubleGeometricTree:
    def __init__(
        self,
        *,
        vertices,
        edges,
        vertex_type,
        barycenter
    ):
        self.vertices = vertices
        self.edges = edges
        self.vertex_type = vertex_type
        self.barycenter = barycenter

# Make a combinatorial tree depending on some parameters:


def make_double_combinatorial_tree(
    a,
    b,
    *,
    N=3,
    barycenter_box_margin=0.3,
    junction_span_a=0.2,
    junction_span_b=0.2,
    junction_time_gap=0.15,
    leaf_margin=0.10,
    leaf_time=0.05,
    root_time=0.95,
):
    """
    a, b:
        The leaves form an a-by-b rectangular array.

    N:
        Ambient cube dimension.
        coordinate 0     = a-direction,
        coordinate 1     = b-direction,
        coordinate N - 1 = time direction.

    barycenter_box_margin:
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

    if a < 1 or b < 1:
        raise ValueError("a and b must be positive.")

    if N < 3:
        raise ValueError("N must be at least 3.")

    if not 0.0 <= barycenter_box_margin < 0.5:
        raise ValueError("barycenter_box_margin must lie in [0, 1/2).")

    if not 0.0 <= leaf_margin < 0.5:
        raise ValueError("leaf_margin must lie in [0, 1/2).")

    if not 0.0 <= leaf_time < barycenter_box_margin:
        raise ValueError("leaf_time must be below barycenter_box_margin.")

    if not 1.0 - barycenter_box_margin < root_time <= 1.0:
        raise ValueError("root_time must lie above 1 - barycenter_box_margin.")

    e_a = np.zeros(N)
    e_a[0] = 1.0

    e_b = np.zeros(N)
    e_b[1] = 1.0

    e_time = np.zeros(N)
    e_time[-1] = 1.0

    leaf_a_coordinates = _equally_spaced_interior(
        a,
        leaf_margin,
    )

    leaf_b_coordinates = _equally_spaced_interior(
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

    return DoubleCombinatorialTree(
        a = int(a),
        b = int(b),
        N = int(N),
        barycenter_box_margin=float(barycenter_box_margin),
        junction_span_a=float(junction_span_a),
        junction_span_b=float(junction_span_b),
        junction_time_gap=float(junction_time_gap),
        e_a = e_a,
        e_b = e_b,
        e_time = e_time,
        leaves = leaves,
        root = root
    )


# Physical barycenter (because of margin complications (x_bar always lies in the interior box
# [margin, 1-margin]^N.)- in reality I actually bypass a lot of this)


def physical_barycenter(double_combinatorial_tree, x):
    N = double_combinatorial_tree.N
    margin = double_combinatorial_tree.barycenter_box_margin

    x = np.asarray(x, dtype=float)

    if x.shape != (N,):
        raise ValueError(f"x must have shape ({N},), got {x.shape}.")

    if np.any(x < 0.0) or np.any(x > 1.0):
        raise ValueError("x must lie in [0, 1]^N.")

    return margin + (1.0 - 2.0 * margin) * x


# Now we construct the map which sends (x, t) to an affine embedded rooted tree.
# The barycenter is x (essentially) and t governs the type of junction we are looking at.
# No two distinct junctions occupy the same position when t != 1/2. At t = 1/2 there is only one junction: AB.


def make_double_geometric_tree_at(double_combinatorial_tree, x, t):

    if not 0.0 <= t <= 1.0:
        raise ValueError("t must lie in [0, 1].")

    a = double_combinatorial_tree.a
    b = double_combinatorial_tree.b

    e_a = double_combinatorial_tree.e_a
    e_b = double_combinatorial_tree.e_b
    e_time = double_combinatorial_tree.e_time

    span_a = double_combinatorial_tree.junction_span_a
    span_b = double_combinatorial_tree.junction_span_b
    time_gap = double_combinatorial_tree.junction_time_gap

    x_bar = physical_barycenter(double_combinatorial_tree, x)

    vertices = {name: position.copy() for name, position in double_combinatorial_tree.leaves.items()}

    vertices["R"] = double_combinatorial_tree.root.copy()

    vertex_type = {name: "leaf" for name in double_combinatorial_tree.leaves}

    vertex_type["R"] = "root"

    edges = []

    # --------------------------------------------------------
    # t < 1/2:
    #
    # b lower a-junctions A_j, then one upper b-junction B.
    # --------------------------------------------------------
    if t < 0.5:
        collapse = 1.0 - _smoothstep(2.0 * t)

        lower_time_offset = -time_gap / (b + 1)

        upper_time_offset = b * time_gap / (b + 1)

        for j in range(b):
            if b == 1:
                spatial_offset = 0.0
            else:
                spatial_offset = span_b * (j / (b - 1) - 0.5)

            vertices[f"A_{j}"] = x_bar + collapse * (
                spatial_offset * e_b + lower_time_offset * e_time
            )

            vertex_type[f"A_{j}"] = "a"

        vertices["B"] = x_bar + collapse * upper_time_offset * e_time

        vertex_type["B"] = "b"

        for i in range(a):
            for j in range(b):
                edges.append((f"L_{i}_{j}", f"A_{j}"))

        for j in range(b):
            edges.append((f"A_{j}", "B"))

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
                edges.append((f"L_{i}_{j}", "AB"))

        edges.append(("AB", "R"))

    # --------------------------------------------------------
    # t > 1/2:
    #
    # a lower b-junctions B_i, then one upper a-junction A.
    # --------------------------------------------------------
    else:
        collapse = 1.0 - _smoothstep(2.0 * (1.0 - t))

        lower_time_offset = -time_gap / (a + 1)

        upper_time_offset = a * time_gap / (a + 1)

        for i in range(a):
            if a == 1:
                spatial_offset = 0.0
            else:
                spatial_offset = span_a * (i / (a - 1) - 0.5)

            vertices[f"B_{i}"] = x_bar + collapse * (
                spatial_offset * e_a + lower_time_offset * e_time
            )

            vertex_type[f"B_{i}"] = "b"

        vertices["A"] = x_bar + collapse * upper_time_offset * e_time

        vertex_type["A"] = "a"

        for i in range(a):
            for j in range(b):
                edges.append((f"L_{i}_{j}", f"B_{i}"))

        for i in range(a):
            edges.append((f"B_{i}", "A"))

        edges.append(("A", "R"))

    return DoubleGeometricTree(
        vertices = vertices,
        edges = edges,
        vertex_type = vertex_type,
        barycenter = x_bar
    )


# Extra maps to access the junctions and edges


def junctions(double_geometric_tree):

    return [
        (
            name,
            double_geometric_tree.vertices[name],
            double_geometric_tree.vertex_type[name],
        )
        for name in double_geometric_tree.vertices
        if double_geometric_tree.vertex_type[name] in {"a", "b", "ab"}
    ]


def edge_segments(double_geometric_tree):
    return [
        (
            child,
            parent,
            double_geometric_tree.vertices[child],
            double_geometric_tree.vertices[parent],
        )
        for child, parent in double_geometric_tree.edges
    ]


# Basic geometry stuff

# First: a smooth interpolation [0, 1] -> [0, 1].


def _smoothstep(s):
    return s * s * (3.0 - 2.0 * s)


# A function which returns equally spaced points in an interval [margin, 1 - margin]


def _equally_spaced_interior(count, margin):
    if count < 1:
        raise ValueError("count must be positive.")

    if count == 1:
        return np.array([0.5])

    return np.linspace(
        margin,
        1.0 - margin,
        count,
    )
