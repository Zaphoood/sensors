import logging
import time
from typing import List, Literal, Optional, Tuple, Union, cast

import numpy as np

from delaunay import sort_triangle
from util import Triangle, Vector, random_scatter_sphere, shift

logging.basicConfig(format="%(message)s", level=logging.WARN)


def plane_sweep(
    points: List[Vector], sweep_direction: Optional[Vector] = None
) -> List[Triangle]:
    if sweep_direction is None:
        sweep_direction = np.array([0, -1, 0], dtype=np.float64)
    triang_north, boundary_north = _plane_sweep_hemisphere(points, sweep_direction)
    triang_south, boundary_south = _plane_sweep_hemisphere(points, -sweep_direction)
    triang_equator = _stich_hemispheres(points, boundary_north, boundary_south)

    return [*triang_north, *triang_south, *triang_equator]


def _plane_sweep_hemisphere(
    points: List[Vector], sweep_direction: Vector
) -> Tuple[List[Triangle], List[int]]:
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
        if sweep_direction.dot(points[vertex]) > 0:
            logging.info("crossed equator, stop.")
            break
        # Edges of the boundary that are visible from current vertex.
        # Entry i corresponds to edge (boundary_vertices[i], boundary_vertices[(i + 1) % len(boundary_vertices)])
        visible_boundary_edges = np.full((len(boundary_vertices)), True)

        for i, (current_idx, next_idx) in enumerate(
            zip(boundary_vertices, shift(boundary_vertices))
        ):
            midpoint = (points[current_idx] + points[next_idx]) / 2
            midpoint /= np.linalg.norm(midpoint)
            for b0, b1 in zip(boundary_vertices, shift(boundary_vertices)):
                if (current_idx, next_idx) == (b0, b1):
                    continue
                visible_boundary_edges[i] &= not (
                    do_arcs_intersect(
                        points[vertex],
                        points[current_idx],
                        points[b0],
                        points[b1],
                    )
                    or do_arcs_intersect(
                        points[vertex],
                        cast(Vector, midpoint),
                        points[b0],
                        points[b1],
                    )
                    or do_arcs_intersect(
                        points[vertex],
                        points[next_idx],
                        points[b0],
                        points[b1],
                    )
                )

        logging.debug(f"{visible_boundary_edges=}")
        for i, (current_idx, next_idx) in enumerate(
            zip(boundary_vertices, shift(boundary_vertices))
        ):
            if visible_boundary_edges[i]:
                new_triangle = sort_triangle((vertex, current_idx, next_idx))
                logging.info(new_triangle)
                triangulation.append(new_triangle)

        # Modify boundary to include the current vertex and remove those that are no longer boundary vertices
        # The latter are vertices which are visible and have visible neighbours.
        # The array `visible_boundary_edges` will look something like this:
        #   [F ... F T... T F ... F] (case i.)
        # or this:
        #   [T ... T F ... F T ... T] (case ii.)
        # or this:
        #   [T T T] (special case iii.)
        #
        # Store the index of the first visible vertex of the sequence as
        # `t_start` and the last as `t_end`; we want to remove vertices at all
        # indices after `t_start` and before `t_end`
        t_start: Optional[int] = None
        t_end: Optional[int] = None
        for current_idx in range(len(visible_boundary_edges)):
            next_idx = (current_idx + 1) % len(visible_boundary_edges)
            if (
                not visible_boundary_edges[current_idx]
                and visible_boundary_edges[next_idx]
            ):
                t_start = next_idx
            if (
                visible_boundary_edges[current_idx]
                and not visible_boundary_edges[next_idx]
            ):
                t_end = next_idx

        if t_start is None and t_end is None:
            raise RuntimeError("All vertices visible or all invisible :(")  # )
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

    return triangulation, boundary_vertices


def _stich_hemispheres(
    points: List[Vector], boundary_north: List[int], boundary_south: List[int]
) -> List[Triangle]:
    n_north = len(boundary_north)
    n_south = len(boundary_south)
    if n_north < 3 or n_south < 3:
        raise ValueError(
            f"Boundaries must be at least of length 3, but got north {n_south} and south {n_south}"
        )

    # Order clockwise from -pi to pi (or anticlockwise? -- doesn't really matter, as long as they're ordered)
    north_cw = sorted(
        [(idx, np.arctan2(points[idx][2], points[idx][0])) for idx in boundary_north],
        key=lambda t: t[1],
    )
    south_cw = sorted(
        [(idx, np.arctan2(points[idx][2], points[idx][0])) for idx in boundary_south],
        key=lambda t: t[1],
    )

    triangulation: List[Triangle] = []
    p_south = 0
    p_north = 0
    # Store the side on which the first closed loop was formed
    first_closed: Optional[Union[Literal["north"], Literal["south"]]] = None
    while True:
        while north_cw[p_north][1] <= south_cw[p_south][1]:
            triangulation.append(
                (
                    north_cw[p_north][0],
                    north_cw[(p_north + 1) % n_north][0],
                    south_cw[p_south][0],
                )
            )
            p_north += 1
            if p_north == n_north:
                first_closed = "north"
                break
        if first_closed is not None:
            break
        while south_cw[p_south][1] <= north_cw[p_north][1]:
            triangulation.append(
                (
                    south_cw[p_south][0],
                    south_cw[(p_south + 1) % n_south][0],
                    north_cw[p_north][0],
                )
            )
            p_south += 1
            if p_south == n_south:
                first_closed = "south"
                break
        if first_closed is not None:
            break

    if first_closed == "north":
        while p_south < n_south:
            triangulation.append(
                (
                    south_cw[p_south][0],
                    south_cw[(p_south + 1) % n_south][0],
                    north_cw[0][0],
                )
            )
            p_south += 1
    else:
        while p_north < n_north:
            triangulation.append(
                (
                    north_cw[p_north][0],
                    north_cw[(p_north + 1) % n_north][0],
                    south_cw[0][0],
                )
            )
            p_north += 1

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


def test_seed(seed: int, n_points: int) -> None:
    print(f"{seed = }")
    np.random.seed(seed)
    points = random_scatter_sphere(n_points)

    triangulation = plane_sweep(points)
    print(triangulation)


def test_many(n_repeat: int, n_points: int) -> None:
    for i in range(n_repeat):
        print(i, end="\r")
        seed = int(time.time() * 1000) % 2**32
        np.random.seed(seed)
        points = random_scatter_sphere(n_points)

        try:
            plane_sweep(points)
        except AssertionError:
            print(seed)
            break


def main():
    # test_seed(int(time.time()), n_points=20)
    test_many(100, 20)


if __name__ == "__main__":
    main()
