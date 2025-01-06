from typing import Any, Optional, Tuple

import numpy as np

from util import Vector


def do_arcs_intersect(a: Vector, b: Vector, c: Vector, d: Vector) -> bool:
    """Check whether the open arcs (a, b) and (c, d) on the unit sphere intersect.

    Here, 'open' means that if two of the endpoints of two of the arcs
    coincide, they are treated as not intersecting. The case `(a, b) == (c, d)`
    is not allowed and will result in a exception. It is assumed that all input
    vectors actually lie on the unit sphere; if they don't, behavior is
    undefined. Neither of the pairs arcs must have antipodal endpoints, i. e.
    `a != -b and c != -d` must hold.
    """
    for x, y in [(a, b), (c, d)]:
        if np.allclose(x, y):
            raise ValueError(f"Arc endpoints are identical: {x}, {y}")
        if np.allclose(x, -y):
            raise ValueError(f"Arc endpoints are antipodal: {x}, {y}")
    a_coincides = np.allclose(a, c) or np.allclose(a, d)
    b_coincides = np.allclose(b, c) or np.allclose(b, d)
    if a_coincides and b_coincides:
        raise ValueError("Line segments must not be identical nor degenerate")
    if a_coincides or b_coincides:
        # Endpoint intersection is treated as non-intersection
        return False

    # Calculate intersection of the two great circles defined by (a, b) and (c, d)
    norm1 = np.linalg.cross(a, b)
    norm2 = np.linalg.cross(c, d)
    inters = np.linalg.cross(norm1, norm2)
    inters /= np.linalg.norm(inters)

    # Great circles intersect in two antipodal points, therefore we also check `-inters`
    return (_is_in_arc(inters, a, b) and _is_in_arc(inters, c, d)) or (
        _is_in_arc(-inters, a, b) and _is_in_arc(-inters, c, d)
    )


def _is_in_arc(v: Vector, a: Vector, b: Vector) -> bool:
    """Check if `v` is contained in the arc from `a` to `b`. It is assumed that
    all points have norm 1 and lie on the same great circle."""
    dot_ab = a.dot(b)
    midpoint = (a + b) / 2

    return v.dot(a) > dot_ab and v.dot(b) > dot_ab and v.dot(midpoint) > 0


def geodesic_distance(v: Vector, w: Vector) -> float:
    return np.arccos(v.dot(w) / (np.linalg.norm(v) * np.linalg.norm(w)))


def get_circumcircle(
    a: Vector, b: Vector, c: Vector
) -> Optional[Tuple[Vector, Vector, np.floating[Any]]]:
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

    return (center, normal / np.linalg.norm(normal), radius)
