from typing import List, Sequence, Tuple, Union, TypeVar, cast

import numpy as np
import numpy.typing as npt


Vector = npt.NDArray[np.float64]
Color = Union[Tuple[int, int, int], List[int]]


WHITE = [255, 255, 255]
BLACK = [0, 0, 0]
RED = [255, 0, 0]
GREEN = [0, 255, 0]
BLUE = [0, 0, 255]
PINK = [251, 198, 207]

T = TypeVar("T")


def shift(a: Sequence[T], n: int = 1) -> Sequence[T]:
    if len(a) == 0:
        return a

    m = n % len(a)

    return [*a[m:], *a[:m]]


def get_rotation_matrix_xz(theta: float) -> npt.NDArray[np.float64]:
    """Return the matrix with rotates a vector by `theta` about the y-axis"""

    return np.array(
        [
            [np.cos(theta), 0, np.sin(theta)],
            [0, 1, 0],
            [-np.sin(theta), 0, np.cos(theta)],
        ]
    )


def get_rotation_matrix_yz(rho: float) -> npt.NDArray[np.float64]:
    """Return the matrix with rotates a vector by `rho` about the x-axis"""

    return np.array(
        [
            [1, 0, 0],
            [0, np.cos(rho), np.sin(rho)],
            [0, -np.sin(rho), np.cos(rho)],
        ]
    )


def get_rotation_matrix(theta: float, rho: float) -> npt.NDArray[np.float64]:
    """Return the matrix with rotates a vector by `theta` about the y-axis and `rho` about the x-axis"""
    R_xz = get_rotation_matrix_xz(theta)
    R_yz = get_rotation_matrix_yz(rho)

    return R_yz.dot(R_xz)


def normalize_homogeneous(a: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    return cast(npt.NDArray[np.float64], a[..., :-1] / a[..., -1])


def closest_point_on_ray(
    origin: Vector, direction: Vector, point: Vector
) -> Tuple[Vector, float]:
    """Given a ray and a point, return the coordinates of the point on the ray
    which is closest to it, as well as the factor lambda in the equation
    `origin + lambda * direction = closest`. Note that `direction` must be of
    unit length.
    """
    lamda = -(origin - point).dot(direction)

    return origin + lamda * direction, lamda
