#!/usr/bin/env python3
import os
from typing import List, Optional, Tuple, cast

import numpy as np
import numpy.typing as npt

from electrons import simulate_electrons_arr
from util import random_scatter_sphere, save_triangulation
from pathlib import Path


def save_electron_animation(
    n_points: int,
    out_dir: str,
    basename: str = "electrons",
    seed: Optional[int] = None,
    iteration_groups: List[Tuple[int, int]] = [(5, 20), (20, 100)],
) -> None:
    print(f"Creating {out_dir}")
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    if seed is not None:
        np.random.seed(seed)

    points = np.array(random_scatter_sphere(n_points))

    step = 0
    save_simulation_state(points, step, out_dir, basename)

    for n_repeat, n_steps in iteration_groups:
        for _ in range(n_repeat):
            points = cast(
                npt.NDArray[np.float64],
                simulate_electrons_arr(points, n_iterations=n_steps),
            )
            step += n_steps
            save_simulation_state(points, step, out_dir, basename)


def save_simulation_state(
    points: npt.NDArray[np.float64], step: int, out_dir: str, basename: str
) -> None:
    path = os.path.join(out_dir, f"{basename}{step}.txt")
    print(f"Saving to {path}")
    save_triangulation(path, points, [])


def main():
    save_electron_animation(n_points=50, out_dir="examples/electrons", seed=177978238)


if __name__ == "__main__":
    main()
