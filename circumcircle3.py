from typing import List, Optional, Sequence, Tuple, cast

import numpy as np
import pygame

from camera import Camera
from renderer import Drawable, Renderer
from draw import draw_circle3d_z, draw_line3d
from face import Face
from illumination import Illumination, Sun
from input import InputManager
from node import Node
from util import PINK, RED, BoundingBox, Color, Vector, shift


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


class App:
    def __init__(self, screen: pygame.surface.Surface) -> None:
        self.screen = screen
        screen_dimensions = (screen.get_width(), screen.get_height())
        self.camera = Camera(
            np.array([0.8, 0.8, -1.0]),
            yaw=-np.pi / 10,
            pitch=-np.pi / 8,
            focal_length=300,
            sensor_dimensions=screen_dimensions,
        )
        self.illumination = Illumination(Sun(np.array([1, -1, 1]), 1), ambience=0.2)
        self.renderer = Renderer(
            self.screen, self.camera, self.illumination, background_color=PINK
        )

        north_pole = Node(np.array([0, 1, 0]))
        south_pole = Node(np.array([0, -1, 0]))
        equator = [
            Node(np.array([1, 0, 0])),
            Node(np.array([0, 0, 1])),
            Node(np.array([-1, 0, 0])),
            Node(np.array([0, 0, -1])),
        ]
        self.nodes = [north_pole, south_pole, *equator]

        self.faces: List[Face] = []
        for node1, node2 in zip(equator, shift(equator)):
            self.faces.append(Face((node1, node2, north_pole)))
            self.faces.append(Face((node1, node2, south_pole)))

        for node in self.nodes:
            self.renderer.register_drawable(node)
        for face in self.faces:
            self.renderer.register_drawable(face)

        self.circumcircle = Circumcircle([north_pole, equator[0], equator[1]], RED)
        self.renderer.register_drawable(self.circumcircle)

        self.input_manager = InputManager(self.nodes, self.camera)

    def update(self) -> bool:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            self.input_manager.handle_event(event)

        return True

    def draw(self) -> None:
        self.renderer.render(show_fps=True)


def main():
    pygame.init()
    screen_size = (1200, 900)
    screen = pygame.display.set_mode(screen_size)
    pygame.display.set_caption("Circumcircle 3d")
    app = App(screen)
    clock = pygame.time.Clock()

    while True:
        if not app.update():
            break
        clock.tick(60)
        app.draw()
        pygame.display.flip()


if __name__ == "__main__":
    main()
