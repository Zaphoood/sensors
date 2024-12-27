import logging
from collections import defaultdict
from typing import Dict, List, Set, Tuple, cast

import numpy as np

from geometry import geodesic_distance, get_circumcircle
from plane_sweep import do_arcs_intersect
from util import Triangle, Vector, sort_edge, sort_triangle


def get_adjacent_triangles(
    triangles: List[Triangle],
) -> Dict[Tuple[int, int], List[int]]:
    # For each edge, store the vertices with which a triangle is formed
    adjacent_triangles: Dict[Tuple[int, int], List[int]] = defaultdict(
        lambda: cast(List[int], [])
    )

    for triangle in triangles:
        for i in range(3):
            edge = sort_edge((triangle[(i + 1) % 3], triangle[(i + 2) % 3]))
            adjacent_triangles[edge].append(triangle[i])

    return adjacent_triangles


def get_delaunay(points: List[Vector], triangles: List[Triangle]) -> List[Triangle]:
    adjacent_triangles = get_adjacent_triangles(triangles)
    any_flipped = True

    while any_flipped:
        logging.info("--- iteration ---")
        any_flipped = False
        edges = list(adjacent_triangles.keys())
        for edge in edges:
            adjacent_vertices = tuple(adjacent_triangles[edge])
            if len(adjacent_vertices) != 2:
                continue

            if should_flip(points, edge, adjacent_vertices):
                flip_edge(adjacent_triangles, edge)
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
) -> bool:
    """Return True if the `edge` with adjacent triangles `(edge[0], edge[1], adjacent[0])`
    and `(edge[0], edge[1], adjacent[1])` is *not* locally delaunay, by checking whether w lies in the circumcircle
    of xyz"""
    logging.debug(f"Checking edge {edge} for flipping")

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
    logging.debug(f"Center {center}, radius {radius:.3f}")
    logging.debug(f"Epicenter {epicenter}, geodesic radius {geodesic_radius:.3f})")
    logging.debug(f"Geodesic distance of w from center: {geodesic_dist_w}")

    return geodesic_dist_w < geodesic_radius


def flip_edge(
    adjacent_triangles: Dict[Tuple[int, int], List[int]],
    edge: Tuple[int, int],
) -> None:
    adjacent = tuple(adjacent_triangles[edge])
    assert len(adjacent) == 2
    logging.info(f"Flipping {edge} => {adjacent}")
    adjacent_triangles.pop(edge)
    adjacent_triangles[sort_edge(adjacent)] = list(edge)

    a, b = edge
    c, d = adjacent

    # TODO: fix this
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
