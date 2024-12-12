from typing import List, Optional, Sequence, Tuple, cast

import numpy as np
import numpy.typing as npt
import pygame

NODE_SIZE = 0.1

WHITE = [255, 255, 255]
BLACK = [0, 0, 0]
RED = [255, 0, 0]

Vector = npt.NDArray[np.float64]

cos = np.cos
sin = np.sin


def normalize_homogeneous(a: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    return cast(npt.NDArray[np.float64], a[..., :-1] / a[..., -1])


def to_homogeneous(v: Vector) -> Vector:
    return np.concatenate([v, [1]])


def get_rotation_matrix(theta: float, rho: float):
    sin = np.sin
    cos = np.cos

    R_xz = np.array(
        [
            [cos(theta), 0, sin(theta)],
            [0, 1, 0],
            [-sin(theta), 0, cos(theta)],
        ]
    )
    R_yz = np.array(
        [
            [1, 0, 0],
            [0, cos(rho), sin(rho)],
            [0, -sin(rho), cos(rho)],
        ]
    )

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

        t_x = sensor_dimensions[0] / 2
        t_y = sensor_dimensions[1] / 2
        self.calibration_matrix = np.array(
            [
                [self.focal_length, 0, t_x],
                [0, -self.focal_length, t_y],
                [0, 0, 1],
            ]
        )

    @property
    def rotation_matrix(self) -> npt.NDArray[np.float64]:
        """Return matrix which rotates world coordinate system vector in to camera coordinate system vector"""

        return get_rotation_matrix(-self.yaw, -self.pitch)

    def point_to_screen(self, point3d: Vector) -> Vector:
        """Map 3D point to pixel coordinates on camera sensor"""

        # From world to camera coordinate system
        point3d_offset = cast(Vector, point3d - self.position)
        point3d_camera_coords = self.rotation_matrix.dot(point3d_offset)

        # Camera coordinate to pixel on 'image sensor'
        point2d_homo = self.calibration_matrix.dot(point3d_camera_coords)

        return normalize_homogeneous(point2d_homo)

    def move(self, offset: Vector) -> None:
        """Move camera by offset according to view coordinate system"""
        self.position += get_rotation_matrix(self.yaw, self.pitch).dot(offset)

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

    @property
    def x(self):
        return self.position[0]

    @property
    def y(self):
        return self.position[1]

    @property
    def z(self):
        return self.position[2]

    def get_screen_polygon(self, camera: Camera) -> List[Tuple[float, float]]:
        points = []
        for dx, dy in [(-1, -1), (1, -1), (1, 1), (-1, 1)]:
            offset = (np.array([self.size, self.size, 0]) / 2) * np.array([dx, dy, 0])
            corner = self.position + offset
            points.append(camera.point_to_screen(cast(Vector, corner)))

        return points

    def draw(self, screen: pygame.Surface, camera: Camera) -> None:
        pygame.draw.polygon(screen, self.color, self.get_screen_polygon(camera), 1)


def draw_line3d(
    screen: pygame.Surface,
    camera: Camera,
    color,
    start: Vector,
    end: Vector,
    width: int = 1,
) -> None:
    start2d = camera.point_to_screen(start)
    end2d = camera.point_to_screen(end)
    pygame.draw.line(screen, color, tuple(start2d), tuple(end2d), width)


def draw_circle3d(
    screen: pygame.Surface,
    camera: Camera,
    color,
    center: Vector,
    radius: float,
    n_points: int,
    width: int = 1,
) -> None:
    thetas = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    xs = center + radius * np.cos(thetas)
    ys = center + radius * np.sin(thetas)

    for i in range(n_points):
        current = (xs[i], ys[i])
        next = (xs[(i + 1) % n_points], ys[(i + 1) % n_points])
        pygame.draw.line(screen, color, current, next, width)


class Circumcircle:
    def __init__(self) -> None:
        self.nodes = [
            Node(np.array([0, 0, 0])),
            Node(np.array([0, 1, 0])),
            Node(np.array([1, 0, 0])),
        ]

    def handle_event(self, _: pygame.event.Event) -> None:
        pass

    def update(self) -> None:
        pass

    def draw(self, screen: pygame.Surface, camera: Camera) -> None:
        for node in self.nodes:
            node.draw(screen, camera)

        params = self.get_circle_params()
        if params is not None:
            center, _ = params
            center_node = Node(center, color=RED)
            center_node.draw(screen, camera)
            # TODO: draw circle

    def get_circle_params(self) -> Optional[Tuple[Vector, Vector]]:
        """Return `center, radius` of circle through the three nodes, if the
        points aren't collinear. Return `None` if they are.

        Note that `radius` is a vector from the center to a point on the boundary."""

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
        v = np.cross(b - a, c - a)

        A = np.array(
            [
                [b1 - a1, b2 - a2, b3 - a3],
                [c1 - b1, c2 - b2, c3 - b3],
                v,
            ]
        )
        rhs = np.array([0.5 * (b_norm2 - a_norm2), 0.5 * (c_norm2 - b_norm2), a.dot(v)])

        try:
            center = np.linalg.solve(A, rhs)
        except np.linalg.LinAlgError:
            return None

        return cast(Vector, center), cast(Vector, a - center)


class App:
    def __init__(self, screen_size: Tuple[int, int]) -> None:
        self.camera = Camera(
            np.array([0.0, 0.0, -2.0]),
            yaw=0.0,
            pitch=0.0,
            focal_length=200,
            sensor_dimensions=screen_size,
        )
        self.circle = Circumcircle()

        self.move_step = 0.2
        self.rotate_step = np.pi / 20

    def update(self) -> bool:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RIGHT:
                    self.camera.yaw += self.rotate_step
                elif event.key == pygame.K_LEFT:
                    self.camera.yaw -= self.rotate_step
                if event.key == pygame.K_UP:
                    self.camera.pitch += self.rotate_step
                elif event.key == pygame.K_DOWN:
                    self.camera.pitch -= self.rotate_step
                elif event.key == pygame.K_w:
                    self.camera.move(np.array([0, 0, self.rotate_step]))
                elif event.key == pygame.K_a:
                    self.camera.move(np.array([-self.rotate_step, 0, 0]))
                elif event.key == pygame.K_s:
                    self.camera.move(np.array([0, 0, -self.rotate_step]))
                elif event.key == pygame.K_d:
                    self.camera.move(np.array([self.rotate_step, 0, 0]))
                elif event.key == pygame.K_y:
                    self.camera.move(np.array([0, self.rotate_step, 0]))
                elif event.key == pygame.K_e:
                    self.camera.move(np.array([0, -self.rotate_step, 0]))
                elif event.key == pygame.K_r:
                    self.camera.reset_position()
                    self.camera.reset_orientation()

        return True

    def draw(self, screen: pygame.Surface) -> None:
        screen.fill(WHITE)
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
