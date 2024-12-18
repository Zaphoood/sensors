import numpy as np

from util import Vector


class Sun:
    def __init__(self, direction: Vector, brightness: float) -> None:
        self.direction = direction / np.linalg.norm(direction)
        self.brightness = brightness

    def get_surface_illumination(self, surface_normal: Vector) -> float:
        return max(0, -self.direction.dot(surface_normal))


class Illumination:
    def __init__(self, sun: Sun, ambience: float) -> None:
        self.sun = sun
        self.ambience = ambience

    def get_surface_illumination(self, surface_normal: Vector) -> float:
        return self.ambience + self.sun.get_surface_illumination(surface_normal)
