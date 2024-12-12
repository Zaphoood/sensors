from typing import List, Tuple, cast

import numpy as np
import numpy.typing as npt
import pygame

NODE_SIZE = 0.1

WHITE = [255, 255, 255]
BLACK = [0, 0, 0]

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
    ) -> None:
        self.position = position
        self.size = size

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
        pygame.draw.polygon(screen, BLACK, self.get_screen_polygon(camera), 1)


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


class App:
    def __init__(self, screen_size: Tuple[int, int]) -> None:
        self.camera = Camera(
            np.array([0.0, 0.0, -2.0]),
            yaw=0.0,
            pitch=0.0,
            focal_length=200,
            sensor_dimensions=screen_size,
        )
        self.nodes = [
            Node(np.array([0, 0, 0])),
            Node(np.array([0, 0, 1])),
            Node(np.array([0, 1, 0])),
            Node(np.array([1, 0, 0])),
        ]

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
        for node in self.nodes:
            node.draw(screen, self.camera)
        for node in self.nodes[1:]:
            draw_line3d(
                screen, self.camera, BLACK, self.nodes[0].position, node.position
            )


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
