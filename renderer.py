import time
from abc import ABCMeta, abstractmethod
from contextlib import suppress
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pygame

from camera import Camera
from illumination import Illumination
from util import BLACK, BoundingBox, Color

FONT_SIZE = 20


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

        self.frames_this_second = 0
        self.frames_last_second = 0
        self.current_second = 0
        self.font = pygame.font.SysFont("Courier", FONT_SIZE)

    def register_drawable(self, drawable: "Drawable") -> None:
        if drawable not in self.drawables:
            self.drawables.append(drawable)

    def deregister_drawable(self, drawable: "Drawable") -> None:
        # Ignore attempt to remove Drawable that isn't registered
        with suppress(ValueError):
            self.drawables.remove(drawable)

    def render(self, screen: pygame.surface.Surface, show_fps: bool = False) -> None:
        screen_width = screen.get_width()
        screen_height = screen.get_height()

        second = int(time.time())
        if second != self.current_second:
            self.frames_last_second = self.frames_this_second
            self.frames_this_second = 0
            self.current_second = second
        self.frames_this_second += 1

        screen.fill(self.background_color)

        current_buffer = pygame.Surface((screen_width, screen_height))
        z_buffer = pygame.Surface((screen_width, screen_height))
        current_z_buffer = pygame.Surface((screen_width, screen_height))

        screen_arr = pygame.surfarray.pixels2d(screen)
        current_buffer_arr = pygame.surfarray.pixels2d(current_buffer)
        current_z_buffer_arr = pygame.surfarray.pixels2d(current_z_buffer)

        z_buffer_fill = distance_to_z_buffer(float("inf"))
        for drawable in self.drawables:
            current_buffer.fill(0)
            current_z_buffer.fill(z_buffer_fill)
            bounding_boxes = drawable.draw(
                current_buffer, current_z_buffer, self.camera, self.illumination
            )

            if bounding_boxes is None:
                z_buffer_arr = pygame.surfarray.pixels2d(z_buffer)
                visible = current_z_buffer_arr > z_buffer_arr
                del z_buffer_arr
                screen_arr[visible] = current_buffer_arr[visible]
                z_buffer.blit(
                    current_z_buffer,
                    (0, 0),
                    special_flags=pygame.BLEND_MAX,
                )
            else:
                for bounding_box in bounding_boxes:
                    start_x, end_x, start_y, end_y = bounding_box
                    start_x = max(0, start_x)
                    end_x = min(screen_width, end_x)
                    start_y = max(0, start_y)
                    end_y = min(screen_height, end_y)

                    z_buffer_arr = pygame.surfarray.pixels2d(z_buffer)
                    visible = (
                        current_z_buffer_arr[start_x:end_x, start_y:end_y]
                        > z_buffer_arr[start_x:end_x, start_y:end_y]
                    )
                    # Unlock `z_buffer` Surface by deleting array reference
                    del z_buffer_arr

                    screen_arr[start_x:end_x, start_y:end_y][visible] = (
                        current_buffer_arr[start_x:end_x, start_y:end_y][visible]
                    )

                    current_z_buffer_patch = current_z_buffer.subsurface(
                        [start_x, start_y, end_x - start_x, end_y - start_y]
                    )
                    z_buffer.blit(
                        current_z_buffer_patch,
                        (start_x, start_y),
                        special_flags=pygame.BLEND_MAX,
                    )

        # Unlock `screen` Surface
        del screen_arr

        if show_fps:
            self.draw_fps(screen)

    def draw_fps(self, screen: pygame.surface.Surface) -> None:
        screen.blit(
            self.font.render(f"fps: {self.frames_last_second}", True, BLACK),
            (10, 10),
        )


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
