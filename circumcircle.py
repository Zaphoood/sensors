from typing import List, Optional, Sequence, Tuple, cast

import numpy as np
import pygame

from camera import Camera
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
        params = self.get_circle_params()
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

    def get_circle_params(self) -> Optional[Tuple[Vector, Vector, float]]:
        """Return `center, normal, radius` of circle through the three nodes, if the
        points aren't collinear. Return `None` if they are."""

        a = self.nodes[0].position
        b = self.nodes[1].position
        c = self.nodes[2].position
        a1, a2, a3 = a
        b1, b2, b3 = b
        c1, c2, c3 = c
        a_norm2 = np.sum(a**2)
        b_norm2 = np.sum(b**2)
        c_norm2 = np.sum(c**2)

        # Vector orthogonal to the plane spanned by vectors `b - a` and `c - a`
        normal = np.cross(b - a, c - a)

        A = np.array(
            [
                [b1 - a1, b2 - a2, b3 - a3],
                [c1 - b1, c2 - b2, c3 - b3],
                normal,
            ]
        )
        rhs = np.array(
            [0.5 * (b_norm2 - a_norm2), 0.5 * (c_norm2 - b_norm2), a.dot(normal)]
        )

        try:
            center = np.linalg.solve(A, rhs)
        except np.linalg.LinAlgError:
            return None

        radius = np.linalg.norm(center - a)

        return (
            cast(Vector, center),
            cast(Vector, normal / np.linalg.norm(normal)),
            cast(np.float64, radius),
        )
