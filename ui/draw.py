from typing import Any, List, Literal, Optional, Sequence, Tuple, Union, cast

import numpy as np
import numpy.typing as npt
import pygame

from ui.camera import Camera
from renderer import distance_to_z_buffer
from util import BoundingBox, Color, Vector, get_bounding_box_2d


def draw_line3d(
    screen: pygame.Surface,
    camera: Camera,
    color: Color,
    start: Vector,
    end: Vector,
    width: int = 1,
) -> List[BoundingBox]:
    start_pos = camera.world_to_screen(start)
    end_pos = camera.world_to_screen(end)
    if start_pos is not None and end_pos is not None:
        start2d, _ = start_pos
        end2d, _ = end_pos
        pygame.draw.line(screen, color, tuple(start2d), tuple(end2d), width)
        return [get_bounding_box_2d(np.array([start_pos, end_pos]))]

    return []


def draw_line3d_z(
    buffer: pygame.Surface,
    z_buffer: pygame.Surface,
    camera: Camera,
    color: Color,
    start: Vector,
    end: Vector,
    width: int = 1,
) -> BoundingBox:
    start_pos = camera.world_to_screen(start)
    end_pos = camera.world_to_screen(end)
    mid_pos = camera.world_to_screen((start + end) / 2)
    if not (start_pos is None or end_pos is None or mid_pos is None):
        start2d, _ = start_pos
        end2d, _ = end_pos
        _, z = end_pos
        pygame.draw.line(buffer, color, tuple(start2d), tuple(end2d), width)
        pygame.draw.line(
            z_buffer, distance_to_z_buffer(z), tuple(start2d), tuple(end2d), width
        )

        return get_bounding_box_2d(np.array([start2d, end2d]))

    return (0, 0, 0, 0)


def draw_triangle3d_z(
    buffer: pygame.Surface,
    z_buffer: pygame.Surface,
    camera: Camera,
    fill_color: Optional[Color],
    edge_color: Optional[Color],
    points: Union[npt.NDArray[np.float64], Sequence[Vector]],
    edge_width: int = 1,
) -> Optional[BoundingBox]:
    if isinstance(points, np.ndarray):
        assert points.shape == (3, 3)
        points_arr = points
    else:
        assert len(points) == 3
        points_arr = np.array(points)
    if fill_color is None and edge_color is None:
        return None

    nodes_2d = [camera.world_to_screen(node) for node in points_arr]
    center = np.mean(points_arr, axis=0)
    center_2d = camera.world_to_screen(center)
    if any(node_2d is None for node_2d in nodes_2d) or center_2d is None:
        return None
    nodes_2d = cast(Sequence[Tuple[Vector, float]], nodes_2d)
    nodes_2d = [tuple(np.round(node_2d).astype(int)) for node_2d, _ in nodes_2d]
    # TODO: Uniform z-depth, will fail with intersecting polygons
    _, z = center_2d
    z_val = distance_to_z_buffer(z)

    if fill_color is not None:
        pygame.draw.polygon(buffer, fill_color, nodes_2d, 0)
        pygame.draw.polygon(z_buffer, z_val, nodes_2d, 0)
    if edge_color is not None:
        pygame.draw.polygon(buffer, edge_color, nodes_2d, edge_width)
        pygame.draw.polygon(z_buffer, z_val, nodes_2d, edge_width)

    return get_bounding_box_2d(np.array(nodes_2d))


def draw_circle3d(
    screen: pygame.Surface,
    camera: Camera,
    color: Color,
    center: Vector,
    normal: Vector,
    radius: float,
    n_points: int,
    width: int = 1,
) -> None:
    starts, ends = get_3d_circle_points(center, normal, radius, n_points)
    for start, end in zip(starts, ends):
        draw_line3d(screen, camera, color, start, end, width)


def draw_circle3d_z(
    buffer: pygame.Surface,
    z_buffer: pygame.Surface,
    camera: Camera,
    color: Color,
    center: Vector,
    normal: Vector,
    radius: float,
    n_points: int,
    width: int = 1,
) -> Sequence[BoundingBox]:
    starts, ends = get_3d_circle_points(center, normal, radius, n_points)
    bounding_boxes: List[BoundingBox] = []
    for start, end in zip(starts, ends):
        bounding_box = draw_line3d_z(buffer, z_buffer, camera, color, start, end, width)
        bounding_boxes.append(bounding_box)

    return bounding_boxes


def get_3d_circle_points(
    center: Vector,
    normal: Vector,
    radius: float,
    n_points: int,
) -> Tuple[npt.NDArray[np.floating[Any]], npt.NDArray[np.floating[Any]]]:
    n1, n2, n3 = normal
    # Construct two vectors orthogonal to the normal and to each other, to use as basis for circle points
    if np.isclose(n1, 0):
        orth1 = np.array([1.0, 0.0, 0.0])
    elif np.isclose(n2, 0):
        orth1 = np.array([0.0, 1.0, 0.0])
    elif np.isclose(n3, 0):
        orth1 = np.array([0.0, 0.0, 1.0])
    else:
        orth1 = np.array([n2 * n3, -2 * n1 * n3, n1 * n2])
    orth1 /= np.linalg.norm(orth1)
    orth2 = np.cross(normal, orth1)

    angles = np.linspace(0, 2 * np.pi, n_points + 1)
    starts = center + radius * (
        np.cos(angles[:-1, np.newaxis]) * orth1
        + np.sin(angles[:-1, np.newaxis]) * orth2
    )
    ends = center + radius * (
        np.cos(angles[1:, np.newaxis]) * orth1 + np.sin(angles[1:, np.newaxis]) * orth2
    )
    return starts, ends


ArcPointCount = Union[Literal["adaptive"], int]


def draw_arc3d_z(
    buffer: pygame.Surface,
    z_buffer: pygame.Surface,
    camera: Camera,
    color: Color,
    a: Vector,
    b: Vector,
    n_points: ArcPointCount,
    width: int = 1,
) -> List[BoundingBox]:
    starts, ends = get_3d_arc_points(a, b, n_points)
    bounding_boxes: List[BoundingBox] = []
    for start, end in zip(starts, ends):
        bounding_box = draw_line3d_z(buffer, z_buffer, camera, color, start, end, width)
        bounding_boxes.append(bounding_box)

    return bounding_boxes


def get_3d_arc_points(
    a: Vector, b: Vector, n_points: ArcPointCount
) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    normal = np.linalg.cross(a, b)
    a_orth = np.linalg.cross(normal, a)
    a_orth /= np.linalg.norm(a_orth)

    angle = np.arccos(np.dot(a, b))

    if n_points == "adaptive":
        n_points = max(1, int(angle / np.pi * 12))
    angles = np.linspace(0, angle, n_points + 1)
    starts = (
        np.cos(angles[:-1, np.newaxis]) * a + np.sin(angles[:-1, np.newaxis]) * a_orth
    )
    ends = np.cos(angles[1:, np.newaxis]) * a + np.sin(angles[1:, np.newaxis]) * a_orth

    return starts, ends
