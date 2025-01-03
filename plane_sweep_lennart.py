from typing import List, Tuple

import numpy as np

from util import Vector


def less_close(
    *numbers: float,
) -> bool:  # Just like <=, but with floating point imprecisions
    for i in range(len(numbers) - 1):
        if numbers[i] > numbers[i + 1] and not np.isclose(numbers[i], numbers[i + 1]):
            return False
    return True


def do_segments_intersect(AB: Tuple[Vector, Vector], CD: Tuple[Vector, Vector]) -> bool:
    # Expects that A and B are different and that C and D are different
    A, B = AB
    C, D = CD

    normal_AB = np.cross(A, B)
    normal_CD = np.cross(C, D)
    if np.allclose(normal_AB, 0) or np.allclose(normal_CD, 0):
        # A segment between antipodal points is not well-defined,
        # so we never want to add it to a triangulation
        return True

    matrix_AB = np.array(AB).T
    matrix_CD = np.array(CD).T

    line_intersection_point = np.cross(normal_AB, normal_CD)
    if np.allclose(line_intersection_point, 0):
        # The segments are contained in the same great circle
        # Project onto the AB-plane
        ONB_second_basis_vector = B - np.dot(A, B) * A
        ONB_second_basis_vector = ONB_second_basis_vector / np.linalg.norm(
            ONB_second_basis_vector
        )

        coord_matrix = np.linalg.lstsq(
            np.array((A, ONB_second_basis_vector)).T, np.array((B, C, D)).T
        )[0]

        # The points are now two-dimensional points on the unit circle, so they can be represented by angles
        angle_A = 0
        angle_B, angle_C, angle_D = np.arctan2(coord_matrix[1], coord_matrix[0]) % (
            2 * np.pi
        )
        # By choice of the basis vectors, we have 0 < angle_B <= pi

        # Check whether the circular arcs overlap
        angle_C, angle_D = sorted([angle_C, angle_D])

        return less_close(angle_A, angle_C, angle_B) or angle_D - angle_C > np.pi

    line_intersection_point /= np.linalg.norm(line_intersection_point)

    s_AB, t_AB = np.linalg.lstsq(matrix_AB, line_intersection_point)[0]
    s_CD, t_CD = np.linalg.lstsq(matrix_CD, line_intersection_point)[0]

    return (
        less_close(0, s_AB)
        and less_close(0, t_AB)
        and less_close(0, s_CD)
        and less_close(0, t_CD)
    ) or (
        less_close(s_AB, 0)
        and less_close(t_AB, 0)
        and less_close(s_CD, 0)
        and less_close(t_CD, 0)
    )


def plane_sweep(
    S: List[Vector],
    sweep_directions: Tuple[Vector, Vector, Vector] = (
        np.array([1, 0, 0]),
        np.array([0, 1, 0]),
        np.array([0, 0, 1]),
    ),
) -> List[Tuple[int, int]]:

    sorted_vertices = sorted(
        list(range(len(S))),
        key=lambda point: (
            np.dot(S[point], sweep_directions[0]),
            np.dot(S[point], sweep_directions[1]),
            np.dot(S[point], sweep_directions[2]),
        ),
    )

    added_edges = []

    for i, vertex in enumerate(sorted_vertices):
        for previous_vertex in sorted_vertices[:i]:
            intersection_detected = False
            for edge in added_edges[::-1]:
                if (
                    vertex not in edge
                    and previous_vertex not in edge
                    and do_segments_intersect(
                        (S[vertex], S[previous_vertex]), (S[edge[0]], S[edge[1]])
                    )
                ):
                    intersection_detected = True
                    break

            if not intersection_detected:
                added_edges.append((previous_vertex, vertex))

    if len(added_edges) != 3 * len(S) - 6:
        raise Exception(
            f"Something has gone wrong. The triangulation has {len(added_edges)} edges, "
            f"but it should have {3 * len(S) - 6}."
        )

    return added_edges
