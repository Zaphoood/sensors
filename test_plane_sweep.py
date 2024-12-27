import logging
import time

import numpy as np

from geometry import do_arcs_intersect
from plane_sweep import plane_sweep
from util import get_edges, random_scatter_sphere


def test_seed(seed: int, n_points: int) -> None:
    np.random.seed(seed)
    points = random_scatter_sphere(n_points)

    triang = plane_sweep(points)
    edges = get_edges(triang)
    for edge1 in edges:
        for edge2 in edges:
            if edge1 == edge2:
                continue
            if do_arcs_intersect(
                points[edge1[0]],
                points[edge1[1]],
                points[edge2[0]],
                points[edge2[1]],
            ):
                raise RuntimeError(
                    f"Intersecting arcs: {edge1}, {edge2}\n"
                    f"Edge 1: {points[edge1[0]]}, {points[edge1[1]]}\n"
                    f"Edge 2: {points[edge2[0]]}, {points[edge2[1]]}"
                )


def test_many(n_repeat: int, n_points: int) -> None:
    for i in range(n_repeat):
        print(i, end="\r")
        seed = int(time.time() * 1000) % 2**32
        np.random.seed(seed)
        try:
            test_seed(seed, n_points)
        except Exception as e:
            print(e)
            print(f"{seed = }")
            break


def main():
    logging.basicConfig(format="%(message)s", level=logging.WARN)

    # seed = int(time.time())
    seed = 145075935
    test_seed(seed, n_points=20)

    # test_many(100, n_points=20)


if __name__ == "__main__":
    main()
