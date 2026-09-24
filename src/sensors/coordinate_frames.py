import numpy as np


def rotation_z(angle_rad):
    """
    Rotation matrix for rotation around the Z-axis.
    """
    c = np.cos(angle_rad)
    s = np.sin(angle_rad)

    return np.array([
        [c, -s, 0],
        [s,  c, 0],
        [0,  0, 1]
    ])


def rotate_vector_z(vector, angle_rad):
    """
    Rotate a 3D vector around Z-axis.
    """
    R = rotation_z(angle_rad)
    return R @ vector


if __name__ == "__main__":

    # Vector pointing forward in phone frame
    phone_vector = np.array([1.0, 0.0, 0.0])

    # Phone is rotated 30 degrees relative to world
    angle = np.deg2rad(30)

    world_vector = rotate_vector_z(phone_vector, angle)

    print("Phone-frame vector:")
    print(phone_vector)

    print("\nWorld-frame vector:")
    print(world_vector)

    print("\nExpected approximately:")
    print("[0.866, 0.500, 0.000]")