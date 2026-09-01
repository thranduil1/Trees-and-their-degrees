
# Example of the tree family - this is not as useful as the plots but that's okay

from Tree_building.py import *

if __name__ == "__main__":
    family = make_tree_family(
        a=3,
        b=2,
        N=3,
    )

    x = np.array([0.5, 0.5, 0.5])

    tree_left = tree_at(family, x, t=0.0)
    tree_middle = tree_at(family, x, t=0.5)
    tree_right = tree_at(family, x, t=1.0)

    print("Physical barycenter:")
    print(tree_left["barycenter"])

    print("\nJunctions at t = 0:")
    for item in junctions(tree_left):
        print(item)

    print("\nJunctions at t = 1/2:")
    for item in junctions(tree_middle):
        print(item)

    print("\nJunctions at t = 1:")
    for item in junctions(tree_right):
        print(item)
