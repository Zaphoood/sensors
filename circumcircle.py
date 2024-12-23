from typing import List, Optional, Sequence, Tuple, cast

import numpy as np
import pygame

from camera import Camera
from delaunay import get_circumcircle
from draw import draw_circle3d_z
from illumination import Illumination
from node import Node
from renderer import Drawable
from util import BoundingBox, Color, Vector


class Circumcircle(Drawable):
    def __init__(self, nodes: List[Node], color: Color) -> None:
        self.nodes = nodes
        self.color = color

    def draw(
        self,
        buffer: pygame.surface.Surface,
        z_buffer: pygame.surface.Surface,
        camera: Camera,
        illumination: Illumination,
    ) -> Optional[Sequence[BoundingBox]]:
        params = get_circumcircle(
            self.nodes[0].position,
            self.nodes[1].position,
            self.nodes[2].position,
        )
        if params is not None:
            center, normal, radius = params
            # TODO: Would be more efficient not to create a new Node every on every draw
            center_node = Node(center, color=self.color)
            center_node.draw(buffer, z_buffer, camera, illumination)

            return draw_circle3d_z(
                buffer,
                z_buffer,
                camera,
                self.color,
                center,
                normal,
                radius,
                n_points=20,
            )

        return None
