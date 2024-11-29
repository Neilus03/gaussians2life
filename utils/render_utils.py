import torch
import math


def fov2focal(fov, resolution):
    return resolution / (2 * math.tan(fov / 2))


def build_intrinsics(fovx, fovy, width, height):
    focal_x = fov2focal(fovx, width)
    focal_y = fov2focal(fovy, height)
    return torch.tensor(
        [
            [focal_x, 0, width / 2],
            [0, focal_y, height / 2],
            [0, 0, 1],
        ],
        dtype=torch.float,
    )
