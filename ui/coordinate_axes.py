from typing import List, Tuple

import numpy as np
import pygame

from ui.camera import Camera
from ui.draw import draw_line3d_z
from ui.illumination import Illumination
from ui.renderer import Drawable, distance_to_z_buffer
from util import BoundingBox, Color


class CoordinateAxes(Drawable):
    def __init__(
        self, color: Color, text_offset: Tuple[int, int] = (0, -15), font_size: int = 16
    ) -> None:
        self.color = color
        self.text_offset = np.array(text_offset)
        self.font = pygame.font.SysFont("Courier", font_size)

    def draw(
        self,
        buffer: pygame.surface.Surface,
        z_buffer: pygame.surface.Surface,
        camera: Camera,
        illumination: Illumination,
    ) -> List[BoundingBox]:
        origin = np.array([0, 0, 0])
        bounding_boxes = []
        for endpoint, label in zip([[1, 0, 0], [0, 1, 0], [0, 0, 1]], ["x", "y", "z"]):
            endpoint3d = np.array(endpoint)

            line_bounding_box = draw_line3d_z(
                buffer, z_buffer, camera, self.color, origin, endpoint3d
            )

            screen_pos = camera.world_to_screen(endpoint3d * 1.1)
            if screen_pos is not None:
                endpoint2d, z = screen_pos
                text_position = tuple(
                    np.round(endpoint2d + self.text_offset).astype(int)
                )
                label_rendered = self.font.render(label, True, self.color)
                buffer.blit(label_rendered, text_position)
                z_buffer.blit(
                    self.font.render(label, True, distance_to_z_buffer(z)),
                    text_position,
                )
                text_bounding_box = (
                    text_position[0],
                    text_position[0] + label_rendered.get_width(),
                    text_position[1],
                    text_position[1] + label_rendered.get_height(),
                )

                bounding_boxes.append(text_bounding_box)

            bounding_boxes.append(line_bounding_box)

        return bounding_boxes
