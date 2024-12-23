from datetime import datetime
from typing import List, Tuple, cast

import numpy as np
import pygame

from camera import Camera
from delaunay import get_delaunay
from draw import draw_line3d
from face import Face
from illumination import Illumination, Sun
from input import InputManager
from node import Node
from renderer import Renderer
from util import PINK, Color, Triangle, Vector, load_triangulation, save_triangulation


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

        points, triangles = load_triangulation("examples/triangulation.txt")
        self.nodes = [Node(point, label=f"{i}") for i, point in enumerate(points)]
        self.triangles: List[Triangle] = [
            cast(Triangle, tuple(sorted(triangle))) for triangle in triangles
        ]
        self.faces: List[Face] = triangles_to_faces(self.nodes, self.triangles)
        for node in self.nodes:
            self.renderer.register_drawable(node)
        for face in self.faces:
            self.renderer.register_drawable(face)

        self.input_manager = InputManager(
            self.nodes,
            self.faces,
            self.handle_add_face,
            self.camera,
            key_callbacks={
                pygame.K_SPACE: (lambda _: self.run_delaunay()),
                pygame.K_e: (lambda _: self.export_triangulation()),
            },
        )

    def run_delaunay(self) -> None:
        delaunay_triangulation = get_delaunay(
            [node.position for node in self.nodes], self.triangles
        )

        for face in self.faces:
            self.renderer.deregister_drawable(face)

        self.triangles = delaunay_triangulation
        self.faces = triangles_to_faces(self.nodes, self.triangles)
        for face in self.faces:
            self.renderer.register_drawable(face)

    def export_triangulation(self) -> None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = f"triangulation_{timestamp}.txt"

        save_triangulation(path, [node.position for node in self.nodes], self.triangles)
        print(f"Saved current triangulation to '{path}'")

    def handle_add_face(self, face: Face) -> None:
        self.faces.append(face)
        self.renderer.register_drawable(face)

    def update(self) -> bool:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            self.input_manager.handle_event(event)

        return True

    def draw(self) -> None:
        self.renderer.render(show_fps=True)


def triangles_to_faces(nodes: List[Node], triangles: List[Triangle]) -> List[Face]:
    return [Face((nodes[t[0]], nodes[t[1]], nodes[t[2]])) for t in triangles]


def random_scatter_sphere(n: int) -> List[Vector]:
    """Randomly scatter `n` points on the surface of the 2-sphere"""
    points = []
    while len(points) < n:
        point = (np.random.random(3) * 2) - 1
        norm = np.linalg.norm(point)
        if not 0 < norm <= 1:
            continue

        points.append(cast(Vector, point / norm))

    return points


def main():
    pygame.init()
    screen_size = (1200, 900)
    screen = pygame.display.set_mode(screen_size)
    pygame.display.set_caption("Circumcircle 3d")
    app = App(screen)

    while True:
        if not app.update():
            break
        app.draw()
        pygame.display.flip()


if __name__ == "__main__":
    main()
