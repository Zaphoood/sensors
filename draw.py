import numpy as np
import pygame

from util import Vector, Color
from camera import Camera


def draw_line3d(
    screen: pygame.Surface,
    camera: Camera,
    color: Color,
    start: Vector,
    end: Vector,
    width: int = 1,
) -> None:
    start2d = camera.world_to_screen(start)
    end2d = camera.world_to_screen(end)
    if start2d is not None and end2d is not None:
        pygame.draw.line(screen, color, tuple(start2d), tuple(end2d), width)


def draw_circle3d(
    screen: pygame.Surface,
    camera: Camera,
    color: Color,
    center: Vector,
    normal: Vector,
    radius: float,
    n_points: int,
    width: int = 1,
) -> None:
    n1, n2, n3 = normal
    # Construct two vectors orthogonal to the normal and to each other, to use as basis for circle points
    if np.isclose(n1, 0):
        orth1 = np.array([1.0, 0.0, 0.0])
    elif np.isclose(n2, 0):
        orth1 = np.array([0.0, 1.0, 0.0])
    elif np.isclose(n3, 0):
        orth1 = np.array([0.0, 0.0, 1.0])
    else:
        orth1 = np.array([n2 * n3, -2 * n1 * n3, n1 * n2])
    orth1 /= np.linalg.norm(orth1)
    orth2 = np.cross(normal, orth1)

    angles = np.linspace(0, 2 * np.pi, n_points + 1)
    starts = center + radius * (
        np.cos(angles[:-1, np.newaxis]) * orth1
        + np.sin(angles[:-1, np.newaxis]) * orth2
    )
    ends = center + radius * (
        np.cos(angles[1:, np.newaxis]) * orth1 + np.sin(angles[1:, np.newaxis]) * orth2
    )
    for start, end in zip(starts, ends):
        draw_line3d(screen, camera, color, start, end, width)
