import argparse
import logging
import os
from datetime import datetime
from typing import List, Optional

import numpy as np

from arc import Arc
from connected import get_min_connecting_radius
from miniball import get_max_enclosing_radius
from plane_sweep import plane_sweep

os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "hide"
import pygame

from camera import Camera
from delaunay import get_delaunay
from face import Face
from illumination import Illumination, Sun
from input import InputManager
from node import Node
from renderer import Renderer
from util import (
    PINK,
    Edge,
    Triangle,
    load_triangulation,
    random_scatter_sphere,
    save_triangulation,
    sort_edge,
    sort_triangle,
    sort_triangulation,
)


GENERATE_N_POINTS = 20
SEED = 177978238
np.random.seed(SEED)


class App:
    def __init__(
        self, screen: pygame.surface.Surface, triangulation_path: Optional[str]
    ) -> None:
        self.screen = screen
        screen_dimensions = (screen.get_width(), screen.get_height())
        self.camera = Camera(
            np.array([0.8, 0.8, -1.0]),
            yaw=-np.pi / 10,
            pitch=-np.pi / 8,
            focal_length=300,
            sensor_dimensions=screen_dimensions,
        )
        self.illumination = Illumination(Sun(np.array([1, -1, 1]), 1), ambience=0.4)
        self.renderer = Renderer(
            self.screen, self.camera, self.illumination, background_color=PINK
        )

        if triangulation_path is not None:
            points, triangles = load_triangulation(triangulation_path, do_sort=True)
        else:
            print(
                f"No input triangulation provided, picking random sample of {GENERATE_N_POINTS} points"
            )
            points = random_scatter_sphere(GENERATE_N_POINTS)
            triangles = []

        self.nodes = [Node(point, label=f"{i}") for i, point in enumerate(points)]
        self.triangles: List[Triangle] = triangles
        self.faces: List[Face] = triangles_to_faces(self.nodes, self.triangles)

        self.edges: List[Edge] = []
        self.arcs: List[Arc] = []

        for node in self.nodes:
            self.renderer.register_drawable(node)
        for face in self.faces:
            self.renderer.register_drawable(face)

        self.input_manager = InputManager(
            self.nodes,
            self.faces,
            self.handle_add_triangle,
            self.camera,
            key_callbacks={
                pygame.K_SPACE: (lambda _: self.run_delaunay()),
                pygame.K_e: (lambda _: self.export_triangulation()),
                pygame.K_x: (lambda _: self.delete_all()),
                pygame.K_s: (lambda _: self.triangulate_plane_sweep()),
                pygame.K_m: (lambda _: self.miniball()),
                pygame.K_c: (lambda _: self.min_connecting_radius()),
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

    def delete_all(self) -> None:
        for face in self.faces:
            self.renderer.deregister_drawable(face)
        self.faces = []
        self.triangles = []

        for arc in self.arcs:
            self.renderer.deregister_drawable(arc)
        self.arcs = []
        self.edges = []

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

    def miniball(self) -> None:
        if len(self.triangles) == 0:
            print("No triangulation")
            return

        max_enclosing_radius = get_max_enclosing_radius(
            [node.position for node in self.nodes], self.triangles
        )
        print(
            f"Largest min-radius of current triangulation: " f"{max_enclosing_radius}"
        )

    def min_connecting_radius(self) -> None:
        if len(self.nodes) == 0:
            print("No vertices")
            return

        min_connecting_radius, adjacency_matrix = get_min_connecting_radius(
            [node.position for node in self.nodes]
        )
        print(
            f"Minimum radius for the Vietoris-Rips graph to be connected: {min_connecting_radius}"
        )

        for i in range(len(self.nodes)):
            for j in range(i + 1, len(self.nodes)):
                if adjacency_matrix[i, j]:
                    self.add_edge((i, j))

    def export_triangulation(self) -> None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = f"triangulation_{timestamp}.txt"

        save_triangulation(
            path,
            [node.position for node in self.nodes],
            sort_triangulation(self.triangles),
        )
        print(f"Saved current triangulation to '{path}'")

    def handle_add_triangle(self, triangle: Triangle) -> None:
        triangle = sort_triangle(triangle)
        if triangle in self.triangles:
            print(f"WARNING: Refusing to add existing triangle {triangle}")
            return
        self.triangles.append(triangle)
        face = triangle_to_face(self.nodes, triangle)
        self.faces.append(face)
        self.renderer.register_drawable(face)

    def add_edge(self, edge: Edge) -> None:
        """Does *not* check if the edge already exists"""
        edge = sort_edge(edge)

        self.edges.append(edge)
        arc = edge_to_arc(self.nodes, edge)
        self.arcs.append(arc)
        self.renderer.register_drawable(arc)

    def update(self) -> bool:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            self.input_manager.handle_event(event)

        return True

    def draw(self) -> None:
        self.renderer.render(show_fps=True, ignore_depth=True)


def triangles_to_faces(nodes: List[Node], triangles: List[Triangle]) -> List[Face]:
    return [Face((nodes[t[0]], nodes[t[1]], nodes[t[2]])) for t in triangles]


def triangle_to_face(nodes: List[Node], triangle: Triangle) -> Face:
    return Face((nodes[triangle[0]], nodes[triangle[1]], nodes[triangle[2]]))


def edge_to_arc(nodes: List[Node], edge: Edge) -> Arc:
    return Arc((nodes[edge[0]], nodes[edge[1]]))


parser = argparse.ArgumentParser(
    description="Interactive visualization for creating Delaunay triangulations of the sphere"
)
parser.add_argument(
    "--file", type=str, required=False, help="Path to the input triangulation"
)


def main():
    logging.basicConfig(format="%(message)s", level=logging.INFO)

    args = parser.parse_args()
    if args.file is not None and not os.path.isfile(args.file):
        print(f"Error: The file '{args.file}' does not exist or is not a file.")
        return

    pygame.init()
    screen_size = (1200, 900)
    screen = pygame.display.set_mode(screen_size)
    pygame.display.set_caption("Triangulation of sphere surface")
    app = App(screen, args.file)

    while True:
        if not app.update():
            break
        app.draw()
        pygame.display.flip()


if __name__ == "__main__":
    main()
