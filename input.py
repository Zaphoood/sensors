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

        mouse_pos = np.array(pygame.mouse.get_pos())
        grabbed_node2d = self.camera.world_to_screen(grabbed_node.position)
        self.grab_info = InputManager.GrabInfo(
            start_position=grabbed_node.position,
            mouse_offset=np.round(mouse_pos - grabbed_node2d).astype(int),
        )

    def cancel_grab(self, grab_info: GrabInfo) -> None:
        assert self.selected_node is not None
        self.nodes[self.selected_node].position = grab_info.start_position

        self.grab_info = None

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
