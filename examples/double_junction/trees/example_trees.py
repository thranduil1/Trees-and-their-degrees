# Example of the combinatorial trees - this is not as useful as the plots but that's okay
import numpy as np

from mathmodels.double_junction.trees.tree_building import (
    junctions,
    make_double_combinatorial_tree,
    make_double_geometric_tree_at,
)

if __name__ == "__main__":
    double_combinatorial_tree = make_double_combinatorial_tree(
        a=3,
        b=2,
        N=3,
    )

    x = np.array([0.5, 0.5, 0.5])

    double_geometric_tree_left = make_double_geometric_tree_at(double_combinatorial_tree, x, t=0.0)
    double_geometric_tree_middle = make_double_geometric_tree_at(double_combinatorial_tree, x, t=0.5)
    double_geometric_tree_right = make_double_geometric_tree_at(double_combinatorial_tree, x, t=1.0)

    print("Physical barycenter:")
    print(double_geometric_tree_left.barycenter)

    print("\nJunctions at t = 0:")
    for item in junctions(double_geometric_tree_left):
        print(item)

    print("\nJunctions at t = 1/2:")
    for item in junctions(double_geometric_tree_middle):
        print(item)

    print("\nJunctions at t = 1:")
    for item in junctions(double_geometric_tree_right):
        print(item)
