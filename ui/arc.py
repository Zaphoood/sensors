from typing import Optional, Sequence, Tuple

import pygame

from ui.camera import Camera
from draw import draw_arc3d_z, draw_line3d_z
from illumination import Illumination
from ui.node import Node
from ui.renderer import Drawable
from util import BLUE, BoundingBox, Color, DrawMode


class Arc(Drawable):
    def __init__(
        self,
        nodes: Tuple[
            Node,
            Node,
        ],
        edge_color: Color = BLUE,
        draw_mode: DrawMode = "arcs",
    ) -> None:
        self.nodes = nodes
        self.color = edge_color
        self.draw_mode: DrawMode = draw_mode

    def draw(
        self,
        buffer: pygame.surface.Surface,
        z_buffer: pygame.surface.Surface,
        camera: Camera,
        illumination: Illumination,
    ) -> Optional[Sequence[BoundingBox]]:
        if self.draw_mode == "triangle":
            return self._draw_straight(buffer, z_buffer, camera, illumination)
        elif self.draw_mode == "arcs":
            return self._draw_arc(buffer, z_buffer, camera)
        else:
            raise ValueError(
                f"Exhaustive check of draw mode in {self.__class__.__name__}.draw(): '{self.draw_mode}'"
            )

    def _draw_straight(
        self,
        buffer: pygame.surface.Surface,
        z_buffer: pygame.surface.Surface,
        camera: Camera,
        illumination: Illumination,
    ) -> Sequence[BoundingBox]:
        return [
            draw_line3d_z(
                buffer,
                z_buffer,
                camera,
                self.color,
                self.nodes[0].position,
                self.nodes[1].position,
                width=2,
            )
        ]

    def _draw_arc(
        self,
        buffer: pygame.surface.Surface,
        z_buffer: pygame.surface.Surface,
        camera: Camera,
    ) -> Optional[Sequence[BoundingBox]]:
        return draw_arc3d_z(
            buffer,
            z_buffer,
            camera,
            self.color,
            self.nodes[0].position,
            self.nodes[1].position,
            n_points="adaptive",
            width=2,
        )
