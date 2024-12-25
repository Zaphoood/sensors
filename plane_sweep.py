from typing import List, Optional, cast

import numpy as np
from delaunay import sort_triangle
from util import Triangle, Vector, random_scatter_sphere, shift
import logging

logging.basicConfig(format="%(message)s", level=logging.DEBUG)


def plane_sweep(
    points: List[Vector], sweep_direction: Optional[Vector] = None
) -> List[Triangle]:
    if sweep_direction is None:
        sweep_direction = np.array([0, -1, 0], dtype=np.float64)

    # Sort-permutation of `points` by sweep direction.
    points_sorted = sorted(
        list(range(len(points))), key=lambda i: sweep_direction.dot(points[i])
    )
    boundary_vertices = points_sorted[:3]
    triangulation = [(points_sorted[0], points_sorted[1], points_sorted[2])]
    logging.info(triangulation[0])
    for vertex in triangulation[0]:
        logging.debug(f"{vertex=} {points[vertex]}")

    for vertex in points_sorted[3:-1]:
        logging.info("--- iteration ---")
        logging.debug(f"{vertex=} {points[vertex]}")
        logging.debug(f"boundary: {boundary_vertices}")
        # Boundary vertices that are visible from current vertex
        visible_boundary_vertices = np.full((len(boundary_vertices)), True)

        special_case_hidden_edge = None
        for i in range(len(boundary_vertices)):
            for b0 in range(len(boundary_vertices)):
                b1 = (b0 + 1) % len(boundary_vertices)
                visible_boundary_vertices[i] &= not do_arcs_intersect(
                    points[vertex],
                    points[boundary_vertices[i]],
                    points[boundary_vertices[b0]],
                    points[boundary_vertices[b1]],
                )

        for current in range(len(boundary_vertices)):
            next = (current + 1) % len(boundary_vertices)

            if visible_boundary_vertices[current] and visible_boundary_vertices[next]:
                if len(boundary_vertices) == 3 and special_case_hidden_edge is None:
                    third = (next + 1) % len(boundary_vertices)

                    point_current = points[boundary_vertices[current]]
                    point_next = points[boundary_vertices[next]]
                    point_third = points[boundary_vertices[third]]
                    midpoint = cast(Vector, (point_current + point_next) / 2)

                    if do_arcs_intersect(
                        points[vertex], midpoint, point_next, point_third
                    ) or do_arcs_intersect(
                        points[vertex], midpoint, point_third, point_current
                    ):
                        special_case_hidden_edge = [
                            boundary_vertices[current],
                            boundary_vertices[next],
                        ]
                        continue

                new_triangle = sort_triangle(
                    (vertex, boundary_vertices[current], boundary_vertices[next])
                )
                logging.info(new_triangle)
                triangulation.append(new_triangle)

        # Modify boundary to include the current vertex and remove those that are no longer boundary vertices
        # The latter are vertices which are visible and have visible neighbours.
        # The array `visible_boundary_vertices` will look something like this:
        #   [F ... F T... T F ... F] (case i.)
        # or this:
        #   [T ... T F ... F T ... T] (case ii.)
        # or this:
        #   [T T T] (special case iii.)
        # The index of T after F is called `t_start`, the index of T before F `t_end`
        # We want to remove all indices after `t_start` and before `t_end`
        t_start: Optional[int] = None
        t_end: Optional[int] = None
        for current in range(len(visible_boundary_vertices)):
            next = (current + 1) % len(visible_boundary_vertices)
            if (
                not visible_boundary_vertices[current]
                and visible_boundary_vertices[next]
            ):
                t_start = next
            if (
                visible_boundary_vertices[current]
                and not visible_boundary_vertices[next]
            ):
                t_end = current

        if t_start is None and t_end is None:
            assert special_case_hidden_edge is not None
            boundary_vertices = [*special_case_hidden_edge, vertex]
        else:
            assert t_start is not None and t_end is not None
            if t_start < t_end:  # case i.
                boundary_vertices = [
                    *boundary_vertices[: t_start + 1],
                    vertex,
                    *boundary_vertices[t_end:],
                ]
            else:  # case ii.
                boundary_vertices = [*boundary_vertices[t_end : t_start + 1], vertex]

    # Fill in final hole with last vertex
    for current, next in zip(boundary_vertices, shift(boundary_vertices)):
        triangulation.append((points_sorted[-1], current, next))

    return triangulation


def do_arcs_intersect(a: Vector, b: Vector, c: Vector, d: Vector) -> bool:
    """Check whether the open arcs (a, b) and (c, d) on the unit sphere intersect.
    Here, 'open' means that if exactly two of the endpoints of two of the arcs coincide, they are treated as not intersecting

    It is assumed that all input vectors actually lie on the unit sphere; if they don't, behavior is undefined.
    Neither of the pairs (a, b) and (c, d) must be antipodal.
    """
    for x, y in [(a, b), (c, d)]:
        if np.allclose(x, y):
            raise ValueError(f"Arc endpoints are identical: {x}, {y}")
        if np.allclose(x, -y):
            raise ValueError(
                f"Cannot determine arc intersection for antipodal vertices: {x}, {y}"
            )
    a_coincides = np.allclose(a, c) or np.allclose(a, d)
    b_coincides = np.allclose(b, c) or np.allclose(b, d)
    if a_coincides and b_coincides:
        raise ValueError("Line segments must not be identical")
    if a_coincides or b_coincides:
        # Endpoint intersection is treated as no intersection
        return False

    # Calculate intersection of the two great circles defined by (a, b) and (c, d)
    norm1 = np.linalg.cross(a, b)
    norm2 = np.linalg.cross(c, d)
    inters = cast(Vector, np.linalg.cross(norm1, norm2))
    inters /= np.linalg.norm(inters)

    return (_is_in_arc(inters, a, b) and _is_in_arc(inters, c, d)) or (
        _is_in_arc(-inters, a, b) and _is_in_arc(-inters, c, d)
    )


def _is_in_arc(v: Vector, a: Vector, b: Vector) -> bool:
    """Check if `v` is contained in the arc from `a` to `b`. It is assumed that
    all points lie on the same circle of radius 1 with its center at the
    origin"""
    dot_ab = a.dot(b)
    midpoint = (a + b) / 2

    return v.dot(a) > dot_ab and v.dot(b) > dot_ab and v.dot(midpoint) > 0


def _test_with_seed(seed: int, n_points: int) -> None:
    np.random.seed(seed)
    points = random_scatter_sphere(n_points)
    points = [cast(Vector, point * [1, -1, 1]) for point in points]

    triangulation = plane_sweep(points)
    print(triangulation)


def main():
    # n_repeat = 10
    # n_points = 20
    # for _ in range(n_repeat):
    #     seed = int(time.time() * 1000) % 2**32

    #     print(f"{seed=}")
    #     np.random.seed(seed)
    #     points = random_scatter_sphere(n_points)

    #     triangulation = plane_sweep(points)
    #     print(triangulation)

    _test_with_seed(4288346723, n_points=20)


if __name__ == "__main__":
    main()
