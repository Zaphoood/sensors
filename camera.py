from typing import Optional, Tuple, Union, cast

import numpy as np
import numpy.typing as npt

from util import (
    get_rotation_matrix,
    get_rotation_matrix_xz,
    get_rotation_matrix_yz,
    normalize_homogeneous,
    Vector,
)


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

        R_xz = get_rotation_matrix_xz(self.yaw)
        R_yz = get_rotation_matrix_yz(self.pitch)

        return R_xz.dot(R_yz.dot(vector))

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

    def get_view_ray(self, point2d: Union[Tuple[int, int], Vector]) -> Vector:
        """Map a 2D point on the camera sensor to a unit vector pointing in that view direction, expressed in camera coordinates"""
        if isinstance(point2d, np.ndarray):
            assert point2d.shape == (2,)

        ray_camera_coords = np.array(
            [point2d[0] - self.t_x, -(point2d[1] - self.t_y), self.focal_length]
        )
        return cast(Vector, ray_camera_coords / np.linalg.norm(ray_camera_coords))

    def get_view_ray_world(self, point2d: Union[Tuple[int, int], Vector]) -> Vector:
        """Map a 2D point on the camera sensor to a unit vector pointing in that view direction, expressed in world coordinates"""
        return self.from_camera_coords(self.get_view_ray(point2d))

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
