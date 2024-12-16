from typing import List, Optional, Sequence, cast

import numpy as np
import pygame

from util import Vector, BLACK
from camera import Camera

NODE_SIZE = 0.1


class Node:
    def __init__(
        self,
        position: Vector,
        # World size in meters
        size: float = NODE_SIZE,
        color: Sequence[int] = BLACK,
    ) -> None:
        self.position = position
        self.size = size
        self.color = color
        self.selected = False

    @property
    def x(self):
        return self.position[0]

    @property
    def y(self):
        return self.position[1]

    @property
    def z(self):
        return self.position[2]

    def get_screen_polygon(self, camera: Camera) -> Optional[List[Vector]]:
        points: List[Vector] = []
        for dx, dy in [(-1, -1), (1, -1), (1, 1), (-1, 1)]:
            offset = (np.array([self.size, self.size, 0]) / 2) * np.array([dx, dy, 0])
            corner = self.position + offset
            point2d = camera.world_to_screen(cast(Vector, corner))
            if point2d is None:
                return None
            points.append(point2d)

        return points

    def on_select(self) -> None:
        self.selected = True

    def on_deselect(self) -> None:
        self.selected = False

    def draw(self, screen: pygame.Surface, camera: Camera) -> None:
        screen_polygon = self.get_screen_polygon(camera)
        if screen_polygon:
            pygame.draw.polygon(
                screen,
                self.color,
                [tuple(corner) for corner in screen_polygon],
                0 if self.selected else 1,
            )
