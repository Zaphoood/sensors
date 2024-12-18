from dataclasses import dataclass
from typing import List, Optional, Tuple, cast

import numpy as np
import numpy.typing as npt
import pygame

from node import Node
from util import Vector, closest_point_on_ray
from camera import Camera

NODE_HITBOX = 0.3


class InputManager:
    @dataclass
    class GrabInfo:
        """Stores information needed for moving a grabbed node"""

        # Distance from the camera to the plane in which the grabbed Node can move
        plane_depth: float

        # Position of the grabbed Node before the grab
        position_before: Vector

        # Offset of the grabbed vector from the projection of the initial view ray onto the movement plane
        offset: Vector

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
                    self.cancel_grab(self.grab_info)
            elif event.key == pygame.K_g:
                self.start_grab()

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

    def start_grab(self) -> None:
        if self.selected_node is None:
            return

        grabbed_node = self.nodes[self.selected_node]
        plane_depth = self.camera.to_camera_coords(
            cast(Vector, grabbed_node.position - self.camera.position)
        )[2]

        mouse_pos = np.array(pygame.mouse.get_pos())
        view_ray = self.camera.get_view_ray(mouse_pos)
        view_ray_in_plane = self.camera.position + self.camera.from_camera_coords(
            view_ray * plane_depth / view_ray[2]
        )
        offset = view_ray_in_plane - grabbed_node.position

        self.grab_info = InputManager.GrabInfo(
            plane_depth, grabbed_node.position, cast(Vector, offset)
        )

    def cancel_grab(self, grab_info: GrabInfo) -> None:
        assert self.selected_node is not None
        self.nodes[self.selected_node].position = grab_info.position_before

        self.grab_info = None

    def handle_mouse_select(self, event: pygame.event.Event) -> None:
        ray = self.camera.get_view_ray_world(event.pos)
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

        new_view_ray = self.camera.get_view_ray(new_mouse_pos)
        new_view_ray_in_plane = self.camera.position + self.camera.from_camera_coords(
            new_view_ray * grab_info.plane_depth / new_view_ray[2]
        )
        self.nodes[self.selected_node].position = cast(
            Vector, new_view_ray_in_plane - grab_info.offset
        )

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
