from typing import Any, List, Literal, Sequence, Set, Tuple, TypeVar, Union, cast

import numpy as np
import numpy.typing as npt

Vector = npt.NDArray[np.floating[Any]]
Color = Union[Tuple[int, int, int], List[int]]
BoundingBox = Tuple[int, int, int, int]
Triangle = Tuple[int, int, int]
Edge = Tuple[int, int]


WHITE = [255, 255, 255]
BLACK = [0, 0, 0]
RED = [255, 0, 0]
GREEN = [0, 255, 0]
BLUE = [0, 0, 255]
PINK = [251, 198, 207]

T = TypeVar("T")

DrawMode = Union[Literal["triangle"], Literal["arcs"]]


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


def get_bounding_box_2d(points: npt.NDArray) -> BoundingBox:
    assert points.ndim == 2 and points.shape[1] == 2
    start_x, start_y = np.min(points, axis=0)
    end_x, end_y = np.max(points, axis=0)

    return (
        int(np.floor(start_x)),
        int(np.ceil(end_x)),
        int(np.floor(start_y)),
        int(np.ceil(end_y)),
    )


def save_triangulation(
    path: str,
    points: List[Vector] | npt.NDArray[np.floating],
    triangles: List[Triangle],
) -> None:
    """Save a set of points and its triangulation to a file."""

    with open(path, "w") as file:
        for point in points:
            file.write(" ".join(str(el) for el in point) + "\n")
        file.write("\n")
        for triangle in triangles:
            file.write(" ".join(str(el) for el in triangle) + "\n")


def load_triangulation(
    path: str, do_sort: bool = False
) -> Tuple[List[Vector], List[Triangle]]:
    reading_points = True
    points: List[Vector] = []
    triangles: List[Tuple[int, int, int]] = []

    with open(path, "r") as file:
        for line in file.readlines():
            line = line.strip()
            if reading_points:
                if len(line) == 0:
                    reading_points = False
                    continue

                point = np.fromstring(line, sep=" ")
                if point.shape != (3,):
                    raise ValueError(f"Vector must be of length 3 but got: '{line}'")
                points.append(point)

            else:
                triangle = tuple(map(int, line.split()))
                if len(triangle) != 3:
                    raise ValueError(f"Triangle must be of length 3 but got: '{line}'")
                triangles.append(triangle)

    triangles_out = sort_triangulation(triangles) if do_sort else triangles

    return points, triangles_out


def sort_triangulation(triangulation: List[Triangle]) -> List[Triangle]:
    """Sort each triangle individually and then the entire triangulation lexicographically"""
    return sorted(map(sort_triangle, triangulation))


def sort_edge(edge: Tuple[int, int]) -> Tuple[int, int]:
    if edge[0] > edge[1]:
        return (edge[1], edge[0])
    else:
        return edge


def sort_triangle(triangle: Triangle) -> Triangle:
    return cast(Triangle, tuple(sorted(triangle)))


def random_scatter_sphere(n: int) -> List[Vector]:
    """Randomly scatter `n` points on the surface of the 2-sphere"""
    points = []
    while len(points) < n:
        point = np.random.randn(3)
        points.append(point / np.linalg.norm(point))

    return points


def get_edges(
    triangles: List[Triangle],
) -> List[Tuple[int, int]]:
    # For each edge, store the vertices with which a triangle is formed
    edges: Set[Tuple[int, int]] = set()

    for triangle in triangles:
        for i in range(3):
            edge = sort_edge((triangle[(i + 1) % 3], triangle[(i + 2) % 3]))
            edges.add(edge)

    return list(edges)

def _normalize(v: Vector) -> Vector:
    return cast(Vector, v / np.linalg.norm(v))