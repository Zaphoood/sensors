from typing import List, cast

import numpy as np
import numpy.typing as npt

from geometry import geodesic_distance
from util import Vector


def simulate_electrons(
    initial_positions: List[Vector], n_iterations: int
) -> List[Vector]:
    electrons = np.array(initial_positions)
    gamma = 1e-3

    for _ in range(n_iterations):
        new_electrons = np.copy(electrons)
        for i in range(electrons.shape[0]):
            for j in range(electrons.shape[0]):
                if i == j:
                    continue
                dist = geodesic_distance(electrons[i], electrons[j])
                direction = electrons[i] - electrons[j]
                direction /= np.linalg.norm(direction)
                direction -= electrons[i].dot(direction) * electrons[i]
                direction /= np.linalg.norm(direction)

                new_electrons[i] += gamma * direction / dist**2

        electrons = cast(
            npt.NDArray[np.float64],
            new_electrons / np.linalg.norm(new_electrons, axis=1)[:, np.newaxis],
        )

    return list(electrons)
