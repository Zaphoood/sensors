import time
from typing import List
import numpy as np

from delaunay import get_delaunay
import delaunay_scipy
from miniball import get_max_enclosing_radius
import modified_Cech_complex
from plane_sweep import plane_sweep
from util import Vector, random_scatter_sphere


def test_methods() -> None:
    n_points = 20
    points = random_scatter_sphere(n_points)


def get_max_radius_delaunay(points: List[Vector]) -> float:
    triangulation = plane_sweep(points)
    delaunay_triangulation = get_delaunay(points, triangulation)

    return get_max_enclosing_radius(points, delaunay_triangulation)


def get_max_radius_czech(points: List[Vector]) -> float:
    radius = modified_Cech_complex.get_full_triangulation_radius(points)
    assert radius is not None, "Czech complex did not return a maximum enclosing radius"

    return radius


def get_max_radius_scipy(points: List[Vector]) -> float:
    return delaunay_scipy.get_R(points)


def main():
    seed = int(time.time())
    np.random.seed(seed)

    test_methods()


if __name__ == "__main__":
    main()
