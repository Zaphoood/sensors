from typing import Optional, Sequence

import numpy as np
import pygame

from ui.illumination import Illumination
from renderer import Drawable, distance_to_z_buffer
from util import BoundingBox, Vector, BLACK
from ui.camera import Camera

NODE_SIZE_PX = 16


TEXT_OFFSET = [-30, -30]
FONT_SIZE = 16


pygame.font.init()
font = pygame.font.SysFont("Courier", FONT_SIZE)


class Node(Drawable):
    def __init__(
        self,
        position: Vector,
        color: Sequence[int] = BLACK,
        label: Optional[str] = None,
    ) -> None:
        self.position: Vector = position.astype(np.float64)
        self.color = color
        self.selected = False

        self.text_offset = np.array(TEXT_OFFSET)
        self.label = label

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
    ) -> Sequence[BoundingBox]:
        screen_pos = camera.world_to_screen(self.position)
        if screen_pos is None:
            return []
        center, z = screen_pos

        rect = [
            int(np.round(center[0] - NODE_SIZE_PX / 2)),
            int(np.round(center[1] - NODE_SIZE_PX / 2)),
            NODE_SIZE_PX,
            NODE_SIZE_PX,
        ]
        width = 0 if self.selected else 1
        z_buffer_val = distance_to_z_buffer(z)
        pygame.draw.rect(buffer, self.color, rect, width)
        pygame.draw.rect(z_buffer, z_buffer_val, rect, width)

        bounding_box = (rect[0], rect[0] + rect[2], rect[1], rect[1] + rect[3])

        if self.label is not None:
            text_position = tuple(np.round(center + self.text_offset).astype(int))
            label_rendered = font.render(self.label, True, self.color)
            buffer.blit(label_rendered, text_position)
            z_buffer.blit(font.render(self.label, False, z_buffer_val), text_position)

            text_bounding_box = (
                text_position[0],
                text_position[0] + label_rendered.get_width(),
                text_position[1],
                text_position[1] + label_rendered.get_height(),
            )

            return [bounding_box, text_bounding_box]

        return [bounding_box]
