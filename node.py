from typing import Optional, Sequence

import numpy as np
import pygame

from illumination import Illumination
from renderer import Drawable, distance_to_z_buffer
from util import BoundingBox, Vector, BLACK
from camera import Camera

NODE_SIZE_PX = 16


class Node(Drawable):
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

    def draw(
        self,
        buffer: pygame.Surface,
        z_buffer: pygame.surface.Surface,
        camera: Camera,
        illumination: Illumination,
    ) -> Optional[Sequence[BoundingBox]]:
        screen_pos = camera.world_to_screen(self.position)
        if screen_pos is None:
            return None
        center, z = screen_pos

        rect = [
            int(np.round(center[0] - NODE_SIZE_PX / 2)),
            int(np.round(center[1] - NODE_SIZE_PX / 2)),
            NODE_SIZE_PX,
            NODE_SIZE_PX,
        ]
        width = 0 if self.selected else 1
        pygame.draw.rect(buffer, self.color, rect, width)
        pygame.draw.rect(z_buffer, distance_to_z_buffer(z), rect, width)

        bounding_box = (rect[0], rect[0] + rect[2], rect[1], rect[1] + rect[3])
        return [bounding_box]
