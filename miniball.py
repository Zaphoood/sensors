from typing import List, Tuple, cast

import numpy as np

from delaunay import get_circumcircle
from util import Triangle, Vector, load_triangulation


def miniball3(a: Vector, b: Vector, c: Vector) -> Tuple[Vector, float]:
    """Given three points on the sphere, return the epicenter of the smallest enclosing circle as well as the radius in the
    geodesic metric"""
    points = [a, b, c]
    for p in points:
        assert np.isclose(np.linalg.norm(p), 1)

    for i in range(3):
        midpoint = (points[i] + points[(i + 1) % 3]) / 2
        midpoint /= np.linalg.norm(midpoint)
        cos_radius = points[i].dot(midpoint)
        if points[(i + 2) % 3].dot(midpoint) >= cos_radius:
            # Points `i` and `(i+1)%3` on boundary; `(i+2)%3` inside
            return cast(Vector, midpoint), np.arccos(cos_radius)

    params = get_circumcircle(a, b, c)
    assert params is not None  # At this point, the points cannot be collinear
    center, _, _ = params
    epicenter = cast(Vector, center / np.linalg.norm(center))

    return epicenter, np.arccos(a.dot(epicenter))


def get_max_enclosing_radius(
    points: List[Vector], triangulation: List[Triangle]
) -> float:
    enclosing_circles = [
        miniball3(points[t[0]], points[t[1]], points[t[2]]) for t in triangulation
    ]
    largest_circle = max(enclosing_circles, key=lambda c: c[1])

    return largest_circle[1]


def main():
    p1, t1 = load_triangulation("examples/triangulation.txt")
    p2, t2 = load_triangulation("examples/triangulation_delaunay.txt")

    print(get_max_enclosing_radius(p1, t1))
    print(get_max_enclosing_radius(p2, t2))


if __name__ == "__main__":
    main()
