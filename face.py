from typing import Optional, Sequence, Tuple, cast

import numpy as np
import pygame

from camera import Camera
from illumination import Illumination
from draw import draw_line3d
from node import Node
from util import BLACK, GREEN, WHITE, Color, Vector


class Face:
    def __init__(
        self,
        nodes: Tuple[
            Node,
            Node,
            Node,
        ],
        edge_color: Color = BLACK,
    ) -> None:
        self.nodes = nodes
        self.edge_color = edge_color

    def get_normal_to_camera(self, camera: Camera) -> Vector:
        normal = np.cross(
            self.nodes[1].position - self.nodes[0].position,
            self.nodes[2].position - self.nodes[0].position,
        )
        normal *= np.sign((camera.position - self.nodes[1].position).dot(normal))
        return normal

    def draw(
        self,
        screen: pygame.surface.Surface,
        camera: Camera,
        illumination: Illumination,
        normal_length: Optional[float] = None,
    ) -> None:
        normal_to_camera = self.get_normal_to_camera(camera)
        illumination_level = illumination.get_surface_illumination(normal_to_camera)
        color = list(
            np.round(np.clip(illumination_level * np.array(WHITE), 0, 255)).astype(int)
        )

        nodes_3d = np.array([node.position for node in self.nodes])
        nodes_2d = [camera.world_to_screen(node) for node in nodes_3d]
        if any(node_2d is None for node_2d in nodes_2d):
            return
        nodes_2d = cast(Sequence[Vector], nodes_2d)
        nodes_2d = [tuple(np.round(node_2d).astype(int)) for node_2d in nodes_2d]

        pygame.draw.polygon(screen, color, nodes_2d, 0)
        pygame.draw.polygon(screen, self.edge_color, nodes_2d, 1)

        if normal_length is not None:
            center = np.mean(nodes_3d, axis=0)
            draw_line3d(
                screen, camera, GREEN, center, center + normal_length * normal_to_camera
            )
