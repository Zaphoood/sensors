#!/usr/bin/env python3
import os
from pathlib import Path
from typing import Any, List, Optional

import numpy as np
import numpy.typing as npt

from delaunay import edge_flip_iterative
from plane_sweep import plane_sweep
from util import Triangle, random_scatter_sphere, save_triangulation, sort_triangulation


def save_delaunay_animation(
    n_points: int,
    out_dir: str,
    basename: str = "delaunay",
    seed: Optional[int] = None,
) -> None:
    print(f"Creating {out_dir}")
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    if seed is not None:
        np.random.seed(seed)

    points = np.array(random_scatter_sphere(n_points))

    triangulation = plane_sweep(list(points))
    save_simulation_state(points, triangulation, 0, out_dir, basename)

    delaunay_generator = edge_flip_iterative(list(points), triangulation)

    for step, triangulation_state in enumerate(delaunay_generator, start=1):
        save_simulation_state(points, triangulation_state, step, out_dir, basename)


def save_simulation_state(
    points: npt.NDArray[np.floating[Any]],
    triangulation: List[Triangle],
    step: int,
    out_dir: str,
    basename: str,
) -> None:
    path = os.path.join(out_dir, f"{basename}{step}.txt")
    print(f"Saving to {path}")
    save_triangulation(path, points, sort_triangulation(triangulation))


def main():
    save_delaunay_animation(n_points=50, out_dir="animation/delaunay", seed=177978238)


if __name__ == "__main__":
    main()
