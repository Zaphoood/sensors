from typing import List, Tuple
import networkx

import numpy as np
import numpy.typing as npt

from util import Vector


def get_min_connecting_radius(
    points: List[Vector],
) -> Tuple[float, npt.NDArray[np.bool]]:
    """Return the minimum radius `r` for the Vieotris-Rips complex to be a connected graph,
    as well as the adjacency matrix of that graph.

    All points must lie on the unit sphere."""
    points_arr = np.array(points)
    assert np.allclose(np.linalg.norm(points_arr, axis=1), 1)

    # Need to clip since numeric errors may result in dot product > 1 even if inputs all have norm 1
    dots = np.clip(points_arr @ points_arr.T, -1, 1)
    pairwise_distances = np.arccos(dots)
    distances_sorted = np.sort(pairwise_distances.flatten())

    for distance in distances_sorted:
        adjacency_mat = pairwise_distances <= distance
        graph = networkx.from_numpy_array(adjacency_mat)
        if networkx.is_connected(graph):
            return distance, adjacency_mat

    raise RuntimeError(
        "Unreachable: Vietoris-Rips graph is not connected even for largest distance pairwise distance"
    )


def main():
    points = [
        np.array([1, 0, 0]),
        np.array([np.sqrt(2), np.sqrt(2), 0]),
        np.array([0, 1, 0]),
    ]
    points = [point / np.linalg.norm(point) for point in points]
    print(get_min_connecting_radius(points))


if __name__ == "__main__":
    main()
