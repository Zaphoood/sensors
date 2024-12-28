import logging
import time

import numpy as np

from geometry import do_arcs_intersect
from plane_sweep import plane_sweep
from util import get_edges, random_scatter_sphere


def test_seed(seed: int, n_points: int) -> float:
    print(f"{seed=}")
    np.random.seed(seed)
    # TODO: Make sure test cases are valid, i. e. not all points on one hemisphere
    points = random_scatter_sphere(n_points)

    t_start = time.time()
    triang = plane_sweep(points)
    t_end = time.time()

    edges = get_edges(triang)
    intersections = []
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
                intersections.append((edge1, edge2))
    for edge1, edge2 in intersections:
        print(
            f"Intersecting arcs: {edge1}, {edge2}\n"
            f"Edge 1: {points[edge1[0]]}, {points[edge1[1]]}\n"
            f"Edge 2: {points[edge2[0]]}, {points[edge2[1]]}"
        )
    if len(intersections) > 0:
        raise RuntimeError("Intesecting arcs")

    return t_end - t_start


def test_many(n_repeat: int, n_points: int) -> None:
    times = np.full(n_repeat, np.nan)
    for i in range(n_repeat):
        print(i, end="\r")
        seed = int(time.time() * 1000) % 2**32
        np.random.seed(seed)
        try:
            times[i] = test_seed(seed, n_points)
        except Exception as e:
            print(e)
            print(f"{seed = }")
            break

    avg_time = np.nanmean(times)
    std_time = np.sqrt(np.nanvar(times))
    print(f"Average time: {avg_time*1000:.1f} ms (std. dev {std_time*1000:.1f} ms)")


GOOD_SEEDS = [
    (177975474, 20),
    (177975875, 20),
    (177976197, 20),
    (177976540, 20),
    (177976874, 20),
    (177977213, 20),
    (177977555, 20),
    (177977891, 20),
    # (177978238, 20),
]


def main():
    logging.basicConfig(format="%(message)s", level=logging.WARN)

    # seed = int(time.time())
    for seed, n_points in GOOD_SEEDS:
        test_seed(seed, n_points)

    # test_many(10, n_points=20)


if __name__ == "__main__":
    main()
