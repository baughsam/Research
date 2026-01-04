# Shape.py #Configures shape of cyclinder and ellipsoid
from abc import ABC, abstractmethod
import numpy as np
from dataclasses import dataclass, field


class Shape(ABC):
    @abstractmethod
    def is_inside(self, x, y, z) -> bool:
        pass

@dataclass(frozen=True)
class EllipticalCylinder(Shape):
    axis_a: float
    axis_b: float
    length_c: float
    center: np.ndarray = field(default_factory=lambda: np.array([0, 0, 0]))

    def __post_init__(self):
        self.center.flags.writeable = False #Make test in Shape tests


    def is_inside(self, x, y, z) -> bool:
        # Shift to center
        dx = x - self.center[0]
        dy = y - self.center[1]
        dz = z - self.center[2]

        # Check length condition
        length_check = np.abs(dz) <= self.length_c / 2

        # Check radial condition
        radial_check = (dx / self.axis_a) ** 2 + (dy / self.axis_b) ** 2 <= 1

        return length_check and radial_check
