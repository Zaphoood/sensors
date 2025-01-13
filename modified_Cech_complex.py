from typing import List, Optional, Tuple, Set
import itertools
import networkx as nx


import numpy as np

from util import Vector


def _less_close(
    *numbers: float,
) -> bool:  # Just like <=, but with floating point imprecisions
    for i in range(len(numbers) - 1):
        if numbers[i] > numbers[i + 1] and not np.isclose(numbers[i], numbers[i + 1]):
            return False
    return True


def get_circumradius(A: Vector, B: Vector, C: Vector) -> float:
    center_or_opposite = np.cross(B - A, C - A)
    center_or_opposite /= np.linalg.norm(center_or_opposite)
    if np.allclose(center_or_opposite, 0):
        return np.pi / 2

    return min(
        np.arccos(A.dot(center_or_opposite)), np.arccos(A.dot(-center_or_opposite))
    )


def get_smallest_enclosing_radius(A: Vector, B: Vector, C: Vector) -> float:
    points = [A, B, C]
    for i in range(3):
        midpoint = points[(i + 1) % 3] + points[(i + 2) % 3]
        if np.allclose(midpoint, 0):
            return np.pi / 2
        midpoint /= np.linalg.norm(midpoint)

        dist = np.arccos(midpoint.dot(points[(i + 1) % 3]))

        if _less_close(np.arccos(midpoint.dot(points[i])), dist):
            return dist

    return get_circumradius(A, B, C)


def get_triangle_order(points: List[Vector]) -> List[Set[int]]:
    n = len(points)

    triangles: List[Set[int]] = []
    enclosing_circle_radii: List[float] = []

    for triangle in itertools.combinations(range(n), 3):
        triangles.append(set(triangle))
        enclosing_circle_radii.append(
            get_smallest_enclosing_radius(
                points[triangle[0]], points[triangle[1]], points[triangle[2]]
            )
        )

    return sorted([x for _, x in sorted(zip(enclosing_circle_radii, triangles))])


def add_simplex_and_larger(sx: Set[int], cx: Set[Tuple[int, ...]], n: int) -> None:
    cx.add(tuple(sorted(sx)))
    for i in range(n):
        if i in sx:
            continue

        builds_larger_simplex = True
        for vertex in sx:
            coface = sx.difference({vertex}).union({i})
            if tuple(sorted(coface)) not in cx:
                builds_larger_simplex = False
                break

        if builds_larger_simplex:
            add_simplex_and_larger(sx.union({i}), cx, n)


def get_R(points: List[Vector]) -> Optional[float]:
    n = len(points)
    all_triangles = get_triangle_order(points)

    complex: Set[Tuple[int, ...]] = set()

    graph = nx.Graph()
    graph.add_nodes_from(range(n))
    graph_connected = False

    for triangle in all_triangles:
        add_simplex_and_larger(triangle, complex, n)
        for vertex in triangle:
            graph.add_edge(*tuple(triangle.difference({vertex})))

        if not graph_connected == None and nx.is_connected(graph):
            graph_connected = True

        if graph_connected:
            euler_char = n
            euler_char -= graph.number_of_edges()
            for sx in complex:
                euler_char -= (-1) ** (len(sx))

            if euler_char == 2:
                return get_smallest_enclosing_radius(
                    *[points[vertex] for vertex in triangle]
                )

    return None
