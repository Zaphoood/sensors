from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple, cast

import numpy as np

from util import Triangle, Vector, load_triangulation


def sort_edge(edge: Tuple[int, int]) -> Tuple[int, int]:
    if edge[0] > edge[1]:
        return (edge[1], edge[0])
    else:
        return edge


def sort_triangle(triangle: Triangle) -> Triangle:
    return cast(Triangle, tuple(sorted(triangle)))


def get_adjacent_triangles(
    triangles: List[Triangle],
) -> Dict[Tuple[int, int], List[int]]:
    # For each edge, store the zero, one or two vertices with which a triangle is formed
    adjacent_triangles: Dict[Tuple[int, int], List[int]] = defaultdict(
        lambda: cast(List[int], [])
    )

    for triangle in triangles:
        for i in range(3):
            edge = sort_edge((triangle[(i + 1) % 3], triangle[(i + 2) % 3]))
            adjacent_triangles[edge].append(triangle[i])

    return adjacent_triangles


def get_delaunay(
    points: List[Vector], triangles: List[Triangle], verbose: bool = False
) -> List[Triangle]:
    adjacent_triangles = get_adjacent_triangles(triangles)
    any_flipped = True
    while any_flipped:
        if verbose:
            print("--- Iteration ---")
        any_flipped = False
        edges = list(adjacent_triangles.keys())
        for edge in edges:
            adjacent_vertices = tuple(adjacent_triangles[edge])
            if len(adjacent_vertices) != 2:
                continue

            if should_flip(points, edge, adjacent_vertices, verbose):
                flip_edge(adjacent_triangles, edge, verbose)
                any_flipped = True

    new_triangulation: Set[Triangle] = set()
    for edge, adjacent_vertices in adjacent_triangles.items():
        for vertex in adjacent_vertices:
            new_triangulation.add(sort_triangle((*edge, vertex)))

    return list(new_triangulation)


def should_flip(
    points: List[Vector],
    edge: Tuple[int, int],
    adjacent: Tuple[int, int],
    verbose: bool = False,
) -> bool:
    """Return True if the `edge` with adjacent triangles `(edge[0], edge[1], adjacent[0])`
    and `(edge[0], edge[1], adjacent[1])` is *not* locally delaunay, by checking whether w lies in the circumcircle
    of xyz"""
    if verbose:
        print(f"Checking edge {edge} for flipping")

    x = points[edge[0]]
    y = points[edge[1]]
    z = points[adjacent[0]]
    w = points[adjacent[1]]
    cc_params = get_circumcircle(x, y, z)
    if cc_params is None:
        raise ValueError(f"Points are collinear: {x}, {y}, {z}")

    center, _, radius = cc_params

    geodesic_radius = np.arcsin(radius)
    epicenter = cast(Vector, center / np.linalg.norm(center))
    geodesic_dist_w = geodesic_distance(epicenter, w)
    if verbose:
        print(f"Center {center}, radius {radius:.3f}")
        print(f"Epicenter {epicenter}, geodesic radius {geodesic_radius:.3f})")
    if verbose:
        print(f"Geodesic distance of w from center: {geodesic_dist_w}")

    return geodesic_dist_w < geodesic_radius


def flip_edge(
    adjacent_triangles: Dict[Tuple[int, int], List[int]],
    edge: Tuple[int, int],
    verbose: bool = False,
) -> None:
    adjacent = tuple(adjacent_triangles[edge])
    assert len(adjacent) == 2
    if verbose:
        print(f"Flipping {edge} => {adjacent}")
    adjacent_triangles.pop(edge)
    adjacent_triangles[sort_edge(adjacent)] = list(edge)

    a, b = edge
    c, d = adjacent

    ac = sort_edge((a, c))
    adjacent_triangles[ac].remove(b)
    adjacent_triangles[ac].append(d)

    bc = sort_edge((b, c))
    adjacent_triangles[bc].remove(a)
    adjacent_triangles[bc].append(d)

    ad = sort_edge((a, d))
    adjacent_triangles[ad].remove(b)
    adjacent_triangles[ad].append(c)

    bd = sort_edge((b, d))
    adjacent_triangles[bd].remove(a)
    adjacent_triangles[bd].append(c)


def geodesic_distance(v: Vector, w: Vector) -> float:
    return np.arccos(v.dot(w) / (np.linalg.norm(v) * np.linalg.norm(w)))


def get_circumcircle(
    a: Vector, b: Vector, c: Vector
) -> Optional[Tuple[Vector, Vector, float]]:
    """Return `center, normal, radius` of circle through the three nodes, if the
    points aren't collinear. Return `None` if they are."""

    a1, a2, a3 = a
    b1, b2, b3 = b
    c1, c2, c3 = c
    a_norm2 = np.sum(a**2)
    b_norm2 = np.sum(b**2)
    c_norm2 = np.sum(c**2)

    # Vector orthogonal to the plane spanned by vectors `b - a` and `c - a`
    normal = np.cross(b - a, c - a)

    A = np.array(
        [
            [b1 - a1, b2 - a2, b3 - a3],
            [c1 - b1, c2 - b2, c3 - b3],
            normal,
        ]
    )
    rhs = np.array(
        [0.5 * (b_norm2 - a_norm2), 0.5 * (c_norm2 - b_norm2), a.dot(normal)]
    )

    try:
        center = np.linalg.solve(A, rhs)
    except np.linalg.LinAlgError:
        return None

    radius = np.linalg.norm(center - a)

    return (
        cast(Vector, center),
        cast(Vector, normal / np.linalg.norm(normal)),
        cast(np.float64, radius),
    )
