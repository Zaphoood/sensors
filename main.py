import argparse
import os
from datetime import datetime
from typing import List

import numpy as np

from plane_sweep import plane_sweep

os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "hide"
import pygame

from camera import Camera
from delaunay import get_delaunay, sort_triangle
from face import Face
from illumination import Illumination, Sun
from input import InputManager
from node import Node
from renderer import Renderer
from util import (
    PINK,
    Triangle,
    load_triangulation,
    save_triangulation,
    sort_triangulation,
)


class App:
    def __init__(self, screen: pygame.surface.Surface, triangulation_path: str) -> None:
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

        points, triangles = load_triangulation(triangulation_path, do_sort=True)
        self.nodes = [Node(point, label=f"{i}") for i, point in enumerate(points)]
        self.triangles: List[Triangle] = triangles
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
                pygame.K_x: (lambda _: self.delete_all_triangles()),
                pygame.K_s: (lambda _: self.triangulate_plane_sweep()),
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

    def delete_all_triangles(self) -> None:
        for face in self.faces:
            self.renderer.deregister_drawable(face)
        self.faces = []
        self.triangles = []

    def triangulate_plane_sweep(self) -> None:
        if len(self.triangles) > 0:
            print(
                "Refusing to run plane sweep since there are already triangles. Press 'x' to delete all triangles"
            )
            return

        if len(self.faces) > 0:
            print(
                "WARNING: List of triangles was empty but there were still Faces, deleting them now"
            )
            for face in self.faces:
                self.renderer.deregister_drawable(face)

        self.triangles = plane_sweep([node.position for node in self.nodes])
        self.faces = triangles_to_faces(self.nodes, self.triangles)
        for face in self.faces:
            self.renderer.register_drawable(face)

    def export_triangulation(self) -> None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = f"triangulation_{timestamp}.txt"

        save_triangulation(
            path,
            [node.position for node in self.nodes],
            sort_triangulation(self.triangles),
        )
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


parser = argparse.ArgumentParser(
    description="Interactive visualization for creating Delaunay triangulations of the sphere"
)
parser.add_argument("file", type=str, help="Path to the input triangulation")


def main():
    args = parser.parse_args()
    if not os.path.isfile(args.file):
        print(f"Error: The file '{args.file}' does not exist.")
        return

    pygame.init()
    screen_size = (1200, 900)
    screen = pygame.display.set_mode(screen_size)
    pygame.display.set_caption("Delaunay")
    app = App(screen, args.file)

    while True:
        if not app.update():
            break
        app.draw()
        pygame.display.flip()


if __name__ == "__main__":
    main()
