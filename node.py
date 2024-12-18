from typing import List, Optional, Sequence, cast

import numpy as np
import pygame

from util import Vector, BLACK
from camera import Camera

NODE_SIZE_PX = 16


class Node:
    def __init__(
        self,
        position: Vector,
        color: Sequence[int] = BLACK,
    ) -> None:
        self.position = position.astype(np.float64)
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

    def on_select(self) -> None:
        self.selected = True

    def on_deselect(self) -> None:
        self.selected = False

    def draw(self, screen: pygame.Surface, camera: Camera) -> None:
        center = camera.world_to_screen(self.position)
        if center is None:
            return

        rect = [
            np.round(center[0] - NODE_SIZE_PX / 2),
            np.round(center[1] - NODE_SIZE_PX / 2),
            NODE_SIZE_PX,
            NODE_SIZE_PX,
        ]
        pygame.draw.rect(screen, self.color, rect, 0 if self.selected else 1)
