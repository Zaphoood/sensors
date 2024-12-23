from typing import Tuple, cast

import numpy as np
import pygame

from camera import Camera
from draw import draw_line3d
from util import Color, Vector


class CoordinateAxes:
    def __init__(
        self, color: Color, text_offset: Tuple[int, int] = (0, -15), font_size: int = 16
    ) -> None:
        self.color = color
        self.text_offset = np.array(text_offset)
        self.font = pygame.font.SysFont("Courier", font_size)

    def draw(self, screen: pygame.Surface, camera: Camera) -> None:
        origin = np.array([0, 0, 0])
        for endpoint, label in zip([[1, 0, 0], [0, 1, 0], [0, 0, 1]], ["x", "y", "z"]):
            endpoint3d = np.array(endpoint)

            draw_line3d(screen, camera, self.color, origin, endpoint3d)

            screen_pos = camera.world_to_screen(cast(Vector, endpoint3d * 1.1))
            if screen_pos is not None:
                endpoint2d, _ = screen_pos
                text_position = tuple(
                    np.round(endpoint2d + self.text_offset).astype(int)
                )
                screen.blit(self.font.render(label, True, self.color), text_position)
