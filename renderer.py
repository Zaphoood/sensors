from abc import ABCMeta, abstractmethod
from contextlib import suppress
from typing import List, Optional, Sequence, Tuple
import time

import numpy as np
import pygame

from camera import Camera
from util import RED, BoundingBox, Color
from illumination import Illumination


class Renderer:
    def __init__(
        self,
        camera: Camera,
        illumination: Illumination,
        background_color: Color,
    ) -> None:
        self.camera = camera
        self.illumination = illumination
        self.background_color = background_color

        self.drawables: List[Drawable] = []

    def register_drawable(self, drawable: "Drawable") -> None:
        if drawable not in self.drawables:
            self.drawables.append(drawable)

    def deregister_drawable(self, drawable: "Drawable") -> None:
        # Ignore attempt to remove Drawable that isn't registered
        with suppress(ValueError):
            self.drawables.remove(drawable)

    def render(self, screen: pygame.surface.Surface) -> None:
        current_buffer = pygame.Surface((screen.get_width(), screen.get_height()))
        z_buffer = pygame.Surface((screen.get_width(), screen.get_height()))
        current_z_buffer = pygame.Surface((screen.get_width(), screen.get_height()))

        z_infinity = distance_to_z_value(float("inf"))
        for drawable in self.drawables:
            current_buffer.fill(0)
            current_z_buffer.fill(z_value_to_z_buffer(z_infinity))
            bounding_boxes = drawable.draw(
                current_buffer, current_z_buffer, self.camera, self.illumination
            )

            screen_arr = pygame.surfarray.pixels2d(screen)
            current_buffer_arr = pygame.surfarray.pixels2d(current_buffer)
            z_buffer_arr = pygame.surfarray.pixels2d(z_buffer)
            current_z_buffer_arr = pygame.surfarray.pixels2d(current_z_buffer)

            if bounding_boxes is None:
                visible = current_z_buffer_arr > z_buffer_arr
                z_buffer_arr[visible] = current_z_buffer_arr[visible]
                screen_arr[visible] = current_buffer_arr[visible]
            else:
                for bounding_box in bounding_boxes:
                    start_x, end_x, start_y, end_y = bounding_box
                    visible = (
                        current_z_buffer_arr[start_x:end_x, start_y:end_y]
                        > z_buffer_arr[start_x:end_x, start_y:end_y]
                    )
                    z_buffer_arr[start_x:end_x, start_y:end_y][visible] = (
                        current_z_buffer_arr[start_x:end_x, start_y:end_y][visible]
                    )
                    screen_arr[start_x:end_x, start_y:end_y][visible] = (
                        current_buffer_arr[start_x:end_x, start_y:end_y][visible]
                    )

        z_buffer_arr = pygame.surfarray.pixels2d(z_buffer)
        screen_arr = pygame.surfarray.pixels3d(screen)
        screen_arr[z_buffer_arr == z_infinity] = self.background_color


def distance_to_z_value(distance: float) -> int:
    """Convert distance from camera to value in 8 bit range"""
    return int(np.floor(255 * np.exp(-distance)))


def z_value_to_z_buffer(z_value: int) -> Tuple[int, int, int]:
    return (0, 0, z_value)


def distance_to_z_buffer(distance: float) -> Tuple[int, int, int]:
    """Convert distance from camera to color value to be written into z-buffer"""
    return z_value_to_z_buffer(distance_to_z_value(distance))


class Drawable(metaclass=ABCMeta):
    @abstractmethod
    def draw(
        self,
        buffer: pygame.surface.Surface,
        z_buffer: pygame.surface.Surface,
        camera: Camera,
        illumination: Illumination,
    ) -> Optional[Sequence[BoundingBox]]:
        # Z-buffer is an RGB surface. For now, only write to last (blue) channel.
        # It goes from 0 (farthest) to 255 (closest).
        #
        # Optionally, return a sequence of bounding boxes `(left, right, top, bottom)` of areas that were changed

        raise NotImplementedError()
