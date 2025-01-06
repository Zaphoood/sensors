import numpy as np
import pygame

from ui.camera import Camera
from ui.draw import draw_line3d
from util import Color


class Floor:
    def __init__(self, color: Color, size: int, step: float = 1) -> None:
        self.color = color
        self.size = size
        self.step = step

    def draw(self, screen: pygame.Surface, camera: Camera) -> None:
        xs = np.linspace(
            -self.size * self.step,
            self.size * self.step,
            2 * self.size + 1,
            endpoint=True,
        )

        for x in xs:
            draw_line3d(
                screen,
                camera,
                self.color,
                np.array([x, 0, -self.size * self.step]),
                np.array([x, 0, self.size * self.step]),
            )

            draw_line3d(
                screen,
                camera,
                self.color,
                np.array([-self.size * self.step, 0, x]),
                np.array([self.size * self.step, 0, x]),
            )
