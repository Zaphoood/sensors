from typing import Literal, Optional, Sequence, Tuple, Union, List
from scipy.spatial import SphericalVoronoi

import numpy as np

from util import Triangle, Vector, sort_edge, sort_triangle, _normalize
from modified_Cech_complex import get_smallest_enclosing_radius


def get_delaunay_triangulation(points: list[Vector]) -> List[Tuple[int, int, int]]:
    triangulation: List[Tuple[int, int, int]] = []

    voronoi_diagram = SphericalVoronoi(points)
    voronoi_vertices = voronoi_diagram.vertices
    regions = voronoi_diagram.regions

    for i, vertex in enumerate(voronoi_vertices):
        found_regions = []

        for j, region in enumerate(regions):
            if i in region:
                found_regions.append(j)

        if len(found_regions) < 3:
            raise ValueError("Something went wrong building the triangulation.")
        if len(found_regions) > 3:
            raise NotImplementedError("Four points lie on a circle.")

        triangulation.append(tuple(sorted(found_regions)))

    if len(triangulation) != 2 * len(points) - 4:
        raise ValueError(
            f"Something went wrong. The number of triangles is equal "
            f"to {len(triangulation)}, instead of the expected number {2*len(points) - 4}."
        )

    return triangulation


def get_R(points: List[Vector]) -> float:
    triangles = get_delaunay_triangulation(points)
    enclosing_radii = []

    for triangle in triangles:
        enclosing_radii.append(
            get_smallest_enclosing_radius(
                points[triangle[0]], points[triangle[1]], points[triangle[2]]
            )
        )

    return max(enclosing_radii)
