from typing import Optional, Sequence, Tuple, cast

import numpy as np
import pygame

from camera import Camera
from illumination import Illumination
from draw import draw_line3d, draw_line3d_z, draw_triangle3d_z
from node import Node
from renderer import Drawable, distance_to_z_buffer
from util import BLACK, GREEN, WHITE, Color, Vector


class Face(Drawable):
    def __init__(
        self,
        nodes: Tuple[
            Node,
            Node,
            Node,
        ],
        edge_color: Color = BLACK,
        draw_normals: bool = False,
    ) -> None:
        self.nodes = nodes
        self.edge_color = edge_color
        self.draw_normals = draw_normals

    def get_normal_to_camera(self, camera: Camera) -> Vector:
        normal = np.cross(
            self.nodes[1].position - self.nodes[0].position,
            self.nodes[2].position - self.nodes[0].position,
        )
        normal *= np.sign((camera.position - self.nodes[1].position).dot(normal))
        return normal

    def draw(
        self,
        buffer: pygame.surface.Surface,
        z_buffer: pygame.surface.Surface,
        camera: Camera,
        illumination: Illumination,
    ) -> None:
        normal_to_camera = self.get_normal_to_camera(camera)
        illumination_level = illumination.get_surface_illumination(normal_to_camera)
        color = list(
            np.round(np.clip(illumination_level * np.array(WHITE), 0, 255)).astype(int)
        )

        nodes_3d = np.array([node.position for node in self.nodes])
        draw_triangle3d_z(
            buffer, z_buffer, camera, color, self.edge_color, nodes_3d, edge_width=1
        )

        if self.draw_normals:
            center = np.mean(nodes_3d, axis=0)
            draw_line3d_z(
                buffer, z_buffer, camera, GREEN, center, center + normal_to_camera
            )
