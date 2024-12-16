from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple, Union, cast

import numpy as np
import numpy.typing as npt
import pygame

NODE_SIZE = 0.1
NODE_HITBOX = 0.3

WHITE = [255, 255, 255]
BLACK = [0, 0, 0]
RED = [255, 0, 0]
BLUE = [0, 0, 255]

Vector = npt.NDArray[np.float64]
Color = Union[Tuple[int, int, int], List[int]]

cos = np.cos
sin = np.sin


def normalize_homogeneous(a: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    return cast(npt.NDArray[np.float64], a[..., :-1] / a[..., -1])


def to_homogeneous(v: Vector) -> Vector:
    return np.concatenate([v, [1]])


def get_rotation_matrix_xz(theta: float):
    """Return the matrix with rotates a vector by `theta` about the y-axis"""

    return np.array(
        [
            [np.cos(theta), 0, np.sin(theta)],
            [0, 1, 0],
            [-np.sin(theta), 0, np.cos(theta)],
        ]
    )


def get_rotation_matrix_yz(rho: float):
    """Return the matrix with rotates a vector by `rho` about the x-axis"""

    return np.array(
        [
            [1, 0, 0],
            [0, np.cos(rho), np.sin(rho)],
            [0, -np.sin(rho), np.cos(rho)],
        ]
    )


def get_rotation_matrix(theta: float, rho: float):
    """Return the matrix with rotates a vector by `theta` about the y-axis and `rho` about the x-axis"""
    R_xz = get_rotation_matrix_xz(theta)
    R_yz = get_rotation_matrix_yz(rho)

    return R_yz.dot(R_xz)


class Camera:
    def __init__(
        self,
        position: Vector,
        yaw: float,
        pitch: float,
        focal_length: float,
        sensor_dimensions: Tuple[int, int],
    ):
        self.position = position
        self._initial_position = np.copy(self.position)

        self.yaw = yaw
        self.pitch = pitch
        self._initial_yaw = yaw
        self._initial_pitch = pitch

        self.focal_length = focal_length

        self.t_x = sensor_dimensions[0] / 2
        self.t_y = sensor_dimensions[1] / 2
        self.calibration_matrix = np.array(
            [
                [self.focal_length, 0, self.t_x],
                [0, -self.focal_length, self.t_y],
                [0, 0, 1],
            ]
        )

    def to_camera_coords(self, vector: Vector) -> npt.NDArray[np.float64]:
        """Rotate vector from the world coordinate system to the camera's coordinate system"""

        return get_rotation_matrix(-self.yaw, -self.pitch).dot(vector)

    def from_camera_coords(self, vector: Vector) -> Vector:
        """Rotate vector from the camera's coordinate system to the world coordinate system"""

        return get_rotation_matrix(self.yaw, self.pitch).dot(vector)

    def world_to_screen(self, point3d: Vector) -> Optional[Vector]:
        """Map 3D point to pixel coordinates on camera sensor. Return `None` if the point lies behind the camera."""
        assert point3d.shape == (3,)

        point3d_offset = cast(Vector, point3d - self.position)
        point3d_camera_coords = self.to_camera_coords(point3d_offset)
        if point3d_camera_coords[2] <= 0:
            return None

        # Camera coordinate to pixel on 'image sensor'
        point2d_homo = self.calibration_matrix.dot(point3d_camera_coords)

        return normalize_homogeneous(point2d_homo)

    def screen_to_world(self, point2d: Union[Tuple[int, int], Vector]) -> Vector:
        """Map a 2D point on the camera sensor to a unit vector defining the view ray"""
        if isinstance(point2d, np.ndarray):
            assert point2d.shape == (2,)

        ray_camera_coords = np.array(
            [point2d[0] - self.t_x, -(point2d[1] - self.t_y), self.focal_length]
        )
        ray_world_coords = get_rotation_matrix(self.yaw, self.pitch).dot(
            ray_camera_coords
        )
        return ray_world_coords / np.linalg.norm(ray_world_coords)

    def pan(self, offset: Vector) -> None:
        """Move camera by offset according to view coordinate system"""
        self.position += self.from_camera_coords(offset)

    def change_pitch(self, delta_pitch: float) -> None:
        self.pitch = self.clip_pitch(self.pitch + delta_pitch)

    def clip_pitch(self, pitch: float) -> float:
        return np.clip(pitch, -0.5 * np.pi, 0.5 * np.pi)

    def change_yaw(self, delta_yaw: float) -> None:
        self.yaw += delta_yaw

    def orbit(self, delta_pitch: float, delta_yaw: float) -> None:
        """Orbit camera around orbit center (world origin) by `pitch` and `yaw`"""
        self.position = get_rotation_matrix_xz(
            self.yaw + delta_yaw,
        ).dot(
            get_rotation_matrix_yz(delta_pitch).dot(
                get_rotation_matrix_xz(-self.yaw).dot(self.position)
            )
        )

        self.pitch += delta_pitch
        self.yaw += delta_yaw

    def orbit_from_to(
        self,
        initial_position: Vector,
        initial_pitch: float,
        initial_yaw: float,
        new_pitch: float,
        new_yaw: float,
    ) -> None:
        """Orbit camera around orbit center (world origin), starting from a given initial configuration to some new pitch and yaw"""
        new_pitch = self.clip_pitch(new_pitch)
        self.position = get_rotation_matrix_xz(
            new_yaw,
        ).dot(
            get_rotation_matrix_yz(new_pitch).dot(
                get_rotation_matrix(-initial_yaw, -initial_pitch).dot(initial_position)
            )
        )

        self.pitch = new_pitch
        self.yaw = new_yaw

    def reset_position(self) -> None:
        self.position = np.copy(self._initial_position)

    def reset_orientation(self) -> None:
        self.yaw = self._initial_yaw
        self.pitch = self._initial_pitch


class Node:
    def __init__(
        self,
        position: Vector,
        # World size in meters
        size: float = NODE_SIZE,
        color: Sequence[int] = BLACK,
    ) -> None:
        self.position = position
        self.size = size
        self.color = color
        self.selected = False

    @property
    def x(self):
        return self.position[0]

    @property
    def y(self):
        return self.position[1]

    @property
    def z(self):
        return self.position[2]

    def get_screen_polygon(self, camera: Camera) -> Optional[List[Vector]]:
        points: List[Vector] = []
        for dx, dy in [(-1, -1), (1, -1), (1, 1), (-1, 1)]:
            offset = (np.array([self.size, self.size, 0]) / 2) * np.array([dx, dy, 0])
            corner = self.position + offset
            point2d = camera.world_to_screen(cast(Vector, corner))
            if point2d is None:
                return None
            points.append(point2d)

        return points

    def on_select(self) -> None:
        self.selected = True

    def on_deselect(self) -> None:
        self.selected = False

    def draw(self, screen: pygame.Surface, camera: Camera) -> None:
        screen_polygon = self.get_screen_polygon(camera)
        if screen_polygon:
            pygame.draw.polygon(
                screen,
                self.color,
                [tuple(corner) for corner in screen_polygon],
                0 if self.selected else 1,
            )


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

            endpoint2d = camera.world_to_screen(cast(Vector, endpoint3d * 1.1))
            if endpoint2d is not None:
                text_position = tuple(
                    np.round(endpoint2d + self.text_offset).astype(int)
                )
                screen.blit(self.font.render(label, True, self.color), text_position)


def draw_line3d(
    screen: pygame.Surface,
    camera: Camera,
    color: Color,
    start: Vector,
    end: Vector,
    width: int = 1,
) -> None:
    start2d = camera.world_to_screen(start)
    end2d = camera.world_to_screen(end)
    if start2d is not None and end2d is not None:
        pygame.draw.line(screen, color, tuple(start2d), tuple(end2d), width)


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
    for start, end in zip(starts, ends):
        draw_line3d(screen, camera, color, start, end, width)


class Circumcircle:
    def __init__(self, nodes: List[Node], color: Color) -> None:
        self.nodes = nodes
        self.color = color

    def draw(self, screen: pygame.Surface, camera: Camera) -> None:
        for node in self.nodes:
            node.draw(screen, camera)

        params = self.get_circle_params()
        if params is not None:
            center, normal, radius = params
            center_node = Node(center, color=self.color)
            center_node.draw(screen, camera)
            draw_circle3d(
                screen, camera, self.color, center, normal, radius, n_points=20
            )

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


class InputManager:
    @dataclass
    class GrabInfo:
        """Stores information needed for moving a grabbed node"""

        # The position of the selected Node before the grab
        start_position: Vector
        # Offset of mouse from position of grabbed node on screen
        mouse_offset: npt.NDArray[np.int64]

    @dataclass
    class PanInfo:
        start_position: Vector
        mouse_start: npt.NDArray[np.int64]

    @dataclass
    class OrbitInfo:
        start_position: Vector
        start_pitch: float
        start_yaw: float
        mouse_start: npt.NDArray[np.int64]

    def __init__(self, nodes: List[Node], camera: Camera):
        self.nodes = nodes
        self.selected_node: Optional[int] = None
        self.grab_info: Optional[InputManager.GrabInfo] = None

        self.camera = camera
        self.pan_info: Optional[InputManager.PanInfo] = None
        self.orbit_info: Optional[InputManager.OrbitInfo] = None

        self.camera_move_step = 0.2
        self.camera_rotate_step = np.pi / 20
        self.rotation_factor: float = np.pi / 300

    def handle_event(self, event: pygame.event.Event) -> None:
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_RIGHT:
                if pygame.key.get_mods() & pygame.KMOD_SHIFT:
                    self.camera.orbit(0, -self.camera_rotate_step)
                else:
                    self.camera.change_yaw(self.camera_rotate_step)
            elif event.key == pygame.K_LEFT:
                if pygame.key.get_mods() & pygame.KMOD_SHIFT:
                    self.camera.orbit(0, self.camera_rotate_step)
                else:
                    self.camera.change_yaw(-self.camera_rotate_step)
            if event.key == pygame.K_UP:
                if pygame.key.get_mods() & pygame.KMOD_SHIFT:
                    self.camera.orbit(-self.camera_rotate_step, 0)
                else:
                    self.camera.change_pitch(self.camera_rotate_step)
            elif event.key == pygame.K_DOWN:
                if pygame.key.get_mods() & pygame.KMOD_SHIFT:
                    self.camera.orbit(self.camera_rotate_step, 0)
                else:
                    self.camera.change_pitch(-self.camera_rotate_step)
            elif event.key == pygame.K_w:
                self.camera.pan(np.array([0, 0, self.camera_rotate_step]))
            elif event.key == pygame.K_a:
                self.camera.pan(np.array([-self.camera_rotate_step, 0, 0]))
            elif event.key == pygame.K_s:
                self.camera.pan(np.array([0, 0, -self.camera_rotate_step]))
            elif event.key == pygame.K_d:
                self.camera.pan(np.array([self.camera_rotate_step, 0, 0]))
            elif event.key == pygame.K_y:
                self.camera.pan(np.array([0, self.camera_rotate_step, 0]))
            elif event.key == pygame.K_e:
                self.camera.pan(np.array([0, -self.camera_rotate_step, 0]))
            elif event.key == pygame.K_r:
                if pygame.key.get_mods() & pygame.KMOD_SHIFT:
                    self.camera.position = np.array([0.0, 0.0, -1.0])
                    self.camera.pitch = 0.0
                    self.camera.yaw = 0.0
                else:
                    self.camera.reset_position()
                    self.camera.reset_orientation()
            elif event.key == pygame.K_ESCAPE:
                if self.grab_info is None:
                    self.selected_node = None
                else:
                    assert self.selected_node is not None
                    self.nodes[self.selected_node].position = (
                        self.grab_info.start_position
                    )

                    self.grab_info = None
            elif event.key == pygame.K_g:
                self.start_stop_grab()

        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:
                if self.grab_info is not None:
                    self.grab_info = None
                else:
                    self.handle_mouse_select(event)
            elif event.button == 2:
                if self.grab_info is not None:
                    return
                if pygame.key.get_mods() & pygame.KMOD_SHIFT:
                    self.pan_info = InputManager.PanInfo(
                        start_position=np.copy(self.camera.position),
                        mouse_start=np.array(pygame.mouse.get_pos()),
                    )
                else:
                    self.orbit_info = InputManager.OrbitInfo(
                        start_position=np.copy(self.camera.position),
                        start_pitch=self.camera.pitch,
                        start_yaw=self.camera.yaw,
                        mouse_start=np.array(pygame.mouse.get_pos()),
                    )
            elif event.button in (4, 5):
                self.handle_mouse_scroll(event)

        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button == 2:
                self.pan_info = None
                self.orbit_info = None

        elif event.type == pygame.MOUSEMOTION:
            self.handle_mouse_move(event.pos)

    def start_stop_grab(self) -> None:
        if self.selected_node is None:
            return

        grabbed_node = self.nodes[self.selected_node]

        mouse_pos = np.array(pygame.mouse.get_pos())
        grabbed_node2d = self.camera.world_to_screen(grabbed_node.position)
        self.grab_info = InputManager.GrabInfo(
            start_position=grabbed_node.position,
            mouse_offset=np.round(mouse_pos - grabbed_node2d).astype(int),
        )

    def handle_mouse_select(self, event: pygame.event.Event) -> None:
        ray = self.camera.screen_to_world(event.pos)
        close_candidates = []
        for i, node in enumerate(self.nodes):
            closest_coords, _ = closest_point_on_ray(
                self.camera.position, ray, node.position
            )
            dist_to_node = np.linalg.norm(closest_coords - node.position)
            if dist_to_node <= NODE_HITBOX:
                close_candidates.append((i, dist_to_node))

        if self.selected_node is not None:
            self.nodes[self.selected_node].on_deselect()
        if len(close_candidates) == 0:
            self.selected_node = None
        else:
            self.selected_node = cast(
                int, min(close_candidates, key=lambda el: el[1])[0]
            )
            self.nodes[self.selected_node].on_select()

    def handle_mouse_move(self, new_mouse_pos: Tuple[int, int]) -> None:
        if self.grab_info is not None:
            self.handle_grab_mouse_move(self.grab_info, new_mouse_pos)
        if self.pan_info is not None:
            self.handle_pan_mouse_move(self.pan_info, new_mouse_pos)
        if self.orbit_info is not None:
            self.handle_orbit_mouse_move(self.orbit_info, new_mouse_pos)

    def handle_grab_mouse_move(
        self, grab_info: GrabInfo, new_mouse_pos: Tuple[int, int]
    ) -> None:
        assert self.selected_node is not None

        mouse_pos = np.array(new_mouse_pos, dtype=np.int64)
        new_pos2d = mouse_pos - grab_info.mouse_offset
        new_ray = self.camera.screen_to_world(new_pos2d.astype(np.float64))

        # Solve for the intersection of the ray through the new 2d
        # position and the plane orthogonal to the vector from the
        # camera origin to the position before the grab. To do this, we
        # set up a homogeneous system of equations.
        d1, d2, d3 = new_ray
        o1, o2, o3 = self.camera.position
        A = np.array(
            [
                [d2, -d1, 0, -d2 * o1 + d1 * o2],
                [0, d3, -d2, -d3 * o2 + d2 * o3],
                [
                    *(grab_info.start_position - self.camera.position),
                    -np.dot(
                        grab_info.start_position,
                        grab_info.start_position - self.camera.position,
                    ),
                ],
            ]
        )
        # Solve by eigenvector corresponding to smallest eigenvalue
        _, _, V = np.linalg.svd(A)
        new_pos3d = cast(Vector, V[-1, :-1] / V[-1, -1])

        self.nodes[self.selected_node].position = new_pos3d

    def handle_pan_mouse_move(
        self, pan_info: PanInfo, new_mouse_pos: Tuple[int, int]
    ) -> None:
        mouse_pos = np.array(new_mouse_pos, dtype=np.int64)
        offset2d = mouse_pos - pan_info.mouse_start
        # TODO: this calculation should be a method of the Camera class
        offset3d = (
            np.concatenate([offset2d * [-1, 1], np.zeros(1)]) / self.camera.focal_length
        )
        self.camera.position = pan_info.start_position + self.camera.from_camera_coords(
            offset3d.astype(np.float64)
        )

    def handle_orbit_mouse_move(
        self, orbit_info: OrbitInfo, new_mouse_pos: Tuple[int, int]
    ) -> None:
        mouse_pos = np.array(new_mouse_pos, dtype=np.int64)
        offset2d = mouse_pos - orbit_info.mouse_start
        new_pitch = orbit_info.start_pitch - offset2d[1] * self.rotation_factor
        new_yaw = orbit_info.start_yaw + offset2d[0] * self.rotation_factor

        self.camera.orbit_from_to(
            orbit_info.start_position,
            orbit_info.start_pitch,
            orbit_info.start_yaw,
            new_pitch,
            new_yaw,
        )

    def handle_mouse_scroll(self, event: pygame.event.Event) -> None:
        direction = 1 if event.button == 4 else -1
        offset = self.camera_move_step * np.array([0, 0, direction])
        self.camera.pan(cast(Vector, offset))


def closest_point_on_ray(
    origin: Vector, direction: Vector, point: Vector
) -> Tuple[Vector, float]:
    """Given a ray and a point, return the coordinates of the point on the ray
    which is closest to it, as well as the factor lambda in the equation
    `origin + lambda * direction = closest`. Note that `direction` must be of
    unit length.
    """
    lamda = -(origin - point).dot(direction)

    return origin + lamda * direction, lamda


class App:
    def __init__(self, screen_size: Tuple[int, int]) -> None:
        self.camera = Camera(
            np.array([0.8, 0.8, -1.0]),
            yaw=-np.pi / 10,
            pitch=-np.pi / 8,
            focal_length=200,
            sensor_dimensions=screen_size,
        )
        self.nodes = [
            Node(np.array([0, 0, 0])),
            Node(np.array([0, 1, 1])),
            Node(np.array([1, 0, 0])),
        ]

        self.circle = Circumcircle(self.nodes, RED)
        self.coordinate_axes = CoordinateAxes(BLACK)

        self.input_manager = InputManager(self.nodes, self.camera)

    def update(self) -> bool:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            self.input_manager.handle_event(event)

        return True

    def draw(self, screen: pygame.Surface) -> None:
        screen.fill(WHITE)
        self.coordinate_axes.draw(screen, self.camera)
        self.circle.draw(screen, self.camera)


def main():
    pygame.init()
    screen_size = (800, 600)
    screen = pygame.display.set_mode(screen_size)
    pygame.display.set_caption("Circumcircle 3d")
    app = App(screen_size)
    clock = pygame.time.Clock()

    while True:
        if not app.update():
            break
        clock.tick(60)
        app.draw(screen)
        pygame.display.flip()


if __name__ == "__main__":
    main()
