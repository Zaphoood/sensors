from typing import List

import numpy as np
import numpy.typing as npt

from geometry import geodesic_distance
from util import Vector

DEFAULT_GAMMA = 1e-2


def simulate_electrons(
    initial_positions: List[Vector], n_iterations: int, gamma: float = DEFAULT_GAMMA
) -> List[Vector]:
    return list(
        simulate_electrons_arr(np.array(initial_positions), n_iterations, gamma)
    )


def simulate_electrons_arr(
    electrons: npt.NDArray[np.floating], n_iterations: int, gamma: float = DEFAULT_GAMMA
) -> npt.NDArray[np.floating]:

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

        electrons = new_electrons / np.linalg.norm(new_electrons, axis=1)[:, np.newaxis]

    return electrons
