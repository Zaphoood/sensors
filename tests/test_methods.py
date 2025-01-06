from os.path import abspath, join, dirname
import time
from typing import List
import sys

import numpy as np

sys.path.append(abspath(join(dirname(__file__), "..")))

import modified_Cech_complex
from delaunay import get_delaunay
from miniball import get_max_enclosing_radius
import delaunay_scipy

from util import Vector, random_scatter_sphere


def test_methods(seed: int, verbose: bool = False) -> None:
    if verbose:
        print(f"{seed = }")
    np.random.seed(seed)

    n_points = 20
    n_iterations = 10
    points = random_scatter_sphere(n_points)

    for _ in range(n_iterations):
        r_delaunay = get_max_radius_delaunay(points)
        r_czech = get_max_radius_czech(points)
        r_scipy = get_max_radius_scipy(points)

        assert np.isclose(r_delaunay, r_czech)
        assert np.isclose(r_delaunay, r_scipy)

        if verbose:
            print(f"{r_delaunay=:.6f}")
            print(f"{r_czech=:.6f}")
            print(f"{r_scipy=:.6f}")

    print("Done.")


def get_max_radius_delaunay(points: List[Vector]) -> float:
    delaunay_triangulation = get_delaunay(points)

    return get_max_enclosing_radius(points, delaunay_triangulation)


def get_max_radius_czech(points: List[Vector]) -> float:
    radius = modified_Cech_complex.get_full_triangulation_radius(points)
    assert radius is not None, "Czech complex did not return a maximum enclosing radius"

    return radius


def get_max_radius_scipy(points: List[Vector]) -> float:
    return delaunay_scipy.get_R(points)


def main():
    seed = int(time.time())
    test_methods(seed=seed)


if __name__ == "__main__":
    main()
