import logging
from typing import List, Literal, Optional, Tuple, Union, cast

import numpy as np

from geometry import do_arcs_intersect
from util import Triangle, Vector, shift, sort_triangle


def plane_sweep(
    points: List[Vector], sweep_direction: Optional[Vector] = None
) -> List[Triangle]:
    if sweep_direction is None:
        sweep_direction = np.array([0, -1, 0], dtype=np.float64)
    triang_north, boundary_north = _plane_sweep_hemisphere(points, sweep_direction)
    triang_south, boundary_south = _plane_sweep_hemisphere(points, -sweep_direction)
    triang_equator = _stitch_hemispheres(points, boundary_north, boundary_south)

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

        visible_vertices = np.full((len(boundary_vertices)), True)
        visible_midpoints = np.full((len(boundary_vertices)), True)
        for i, (current, next) in enumerate(
            zip(boundary_vertices, shift(boundary_vertices))
        ):
            midpoint = (points[current] + points[next]) / 2
            midpoint /= np.linalg.norm(midpoint)

            for b0, b1 in zip(boundary_vertices, shift(boundary_vertices)):
                visible_vertices[i] &= (
                    (current == b0) or (current == b1)
                ) or not do_arcs_intersect(
                    points[vertex],
                    points[current],
                    points[b0],
                    points[b1],
                )
                visible_midpoints[i] &= (
                    (current, next) == (b0, b1)
                ) or not do_arcs_intersect(
                    points[vertex],
                    cast(Vector, midpoint),
                    points[b0],
                    points[b1],
                )

        logging.debug(f"{visible_vertices=}")
        logging.debug(f"{visible_midpoints=}")
        # Edges of the boundary that are visible from current vertex.
        # Entry i corresponds to edge (boundary_vertices[i], boundary_vertices[(i + 1) % len(boundary_vertices)])
        visible_boundary_edges = (
            visible_vertices
            & visible_midpoints
            & np.hstack([visible_vertices[1:], visible_vertices[:1]])
        )
        logging.debug(f"{visible_boundary_edges=}")
        for i, (current, next) in enumerate(
            zip(boundary_vertices, shift(boundary_vertices))
        ):
            if visible_boundary_edges[i]:
                new_triangle = sort_triangle((vertex, current, next))
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
        for current in range(len(visible_boundary_edges)):
            next = (current + 1) % len(visible_boundary_edges)
            if not visible_boundary_edges[current] and visible_boundary_edges[next]:
                t_start = next
            if visible_boundary_edges[current] and not visible_boundary_edges[next]:
                t_end = next

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


def _stitch_hemispheres(
    points: List[Vector], boundary_north: List[int], boundary_south: List[int]
) -> List[Triangle]:
    logging.info("--- _stich_hemispheres ---")
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
    north_pole = np.array([0, 1, 0])
    south_pole = np.array([0, -1, 0])
    # Pointers to current vertex on south and north boundary
    ps = 0
    pn = 0
    # Store the side on which the first closed loop was formed
    first_closed: Optional[Union[Literal["north"], Literal["south"]]] = None
    while True:
        while True:
            ps_next = (ps + 1) % n_south
            pn_next = (pn + 1) % n_north
            n = north_cw[pn][0]
            n_next = north_cw[pn_next][0]
            s = south_cw[ps][0]
            s_next = south_cw[ps_next][0]
            if (
                (
                    north_cw[pn][1] > south_cw[ps][1]
                    and clockwise_distance(south_cw[ps][1], north_cw[pn_next][1])
                    > np.pi
                )
                or do_arcs_intersect(
                    points[s], points[n_next], points[s_next], south_pole
                )
                or do_arcs_intersect(points[s], points[n_next], points[n], north_pole)
            ):
                break
            new_triangle = (n, n_next, s)
            logging.info(new_triangle)
            triangulation.append(new_triangle)
            pn += 1
            if pn == n_north:
                first_closed = "north"
                break
        if first_closed is not None:
            break

        while True:
            ps_next = (ps + 1) % n_south
            pn_next = (pn + 1) % n_north
            n = north_cw[pn][0]
            n_next = north_cw[pn_next][0]
            s = south_cw[ps][0]
            s_next = south_cw[ps_next][0]
            if (
                (
                    south_cw[ps][1] > north_cw[pn][1]
                    and clockwise_distance(north_cw[pn][1], south_cw[ps_next][1])
                    > np.pi
                )
                or do_arcs_intersect(
                    points[n],
                    points[s_next],
                    points[n_next],
                    north_pole,
                )
                or do_arcs_intersect(
                    points[n],
                    points[s_next],
                    points[s],
                    south_pole,
                )
            ):
                break
            new_triangle = (s, s_next, n)
            logging.info(new_triangle)
            triangulation.append(new_triangle)
            ps += 1
            if ps == n_south:
                first_closed = "south"
                break
        if first_closed is not None:
            break

    if first_closed == "north":
        while ps < n_south:
            new_triangle = (
                south_cw[ps][0],
                south_cw[(ps + 1) % n_south][0],
                north_cw[0][0],
            )
            logging.info(new_triangle)
            triangulation.append(new_triangle)
            ps += 1
    else:
        while pn < n_north:
            new_triangle = (
                north_cw[pn][0],
                north_cw[(pn + 1) % n_north][0],
                south_cw[0][0],
            )
            logging.info(new_triangle)
            triangulation.append(new_triangle)
            pn += 1

    return triangulation


def clockwise_distance(first: float, second: float) -> float:
    if second >= first:
        return second - first
    else:
        return 2 * np.pi - (first - second)
