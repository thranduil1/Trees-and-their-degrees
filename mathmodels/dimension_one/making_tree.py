import numpy as np

# Construct a tree with one a-junction inside an N-dimensional cube and fixed junction angle.
def make_family_one(
    dimension,
    junction_angle_deg=30.0,
):

    if dimension < 1:
        raise ValueError("dimension must be positive")

    angle_rad = np.deg2rad(junction_angle_deg)

    return {
        "dimension": dimension,
        "junction_angle_rad": angle_rad,
    }
