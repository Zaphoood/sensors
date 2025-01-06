from typing import Optional, Sequence, Tuple

import numpy as np
import pygame

from ui.camera import Camera
from ui.draw import draw_arc3d_z, draw_line3d_z, draw_triangle3d_z
from illumination import Illumination
from ui.node import Node
from ui.renderer import Drawable
from util import BLACK, GREEN, WHITE, BoundingBox, Color, DrawMode, Vector, shift


class Face(Drawable):
    def __init__(
        self,
        nodes: Tuple[
            Node,
            Node,
            Node,
        ],
        edge_color: Color = BLACK,
        draw_mode: DrawMode = "arcs",
        draw_normals: bool = False,
    ) -> None:
        self.nodes = nodes
        self.edge_color = edge_color
        self.draw_mode: DrawMode = draw_mode
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
    ) -> Optional[Sequence[BoundingBox]]:
        if self.draw_mode == "triangle":
            return self._draw_triangle(buffer, z_buffer, camera, illumination)
        elif self.draw_mode == "arcs":
            return self._draw_arcs(buffer, z_buffer, camera)
        else:
            raise ValueError(
                f"Exhaustive check of draw mode in {self.__class__.__name__}.draw(): '{self.draw_mode}'"
            )

    def _draw_triangle(
        self,
        buffer: pygame.surface.Surface,
        z_buffer: pygame.surface.Surface,
        camera: Camera,
        illumination: Illumination,
    ) -> Optional[Sequence[BoundingBox]]:
        normal_to_camera = self.get_normal_to_camera(camera)
        illumination_level = illumination.get_surface_illumination(normal_to_camera)
        color = list(
            np.round(np.clip(illumination_level * np.array(WHITE), 0, 255)).astype(int)
        )

        nodes_3d = np.array([node.position for node in self.nodes])
        bounding_box = draw_triangle3d_z(
            buffer, z_buffer, camera, color, self.edge_color, nodes_3d, edge_width=1
        )

        if self.draw_normals:
            center = np.mean(nodes_3d, axis=0)
            draw_line3d_z(
                buffer, z_buffer, camera, GREEN, center, center + normal_to_camera
            )

        return bounding_box and [bounding_box]

    def _draw_arcs(
        self,
        buffer: pygame.surface.Surface,
        z_buffer: pygame.surface.Surface,
        camera: Camera,
    ) -> Sequence[BoundingBox]:
        bounding_boxes = []
        for node1, node2 in zip(self.nodes, shift(self.nodes)):
            new_bounding_boxes = draw_arc3d_z(
                buffer,
                z_buffer,
                camera,
                self.edge_color,
                node1.position,
                node2.position,
                n_points="adaptive",
            )
            bounding_boxes.extend(new_bounding_boxes)

        return bounding_boxes
