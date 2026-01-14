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
    axis_a: float = 5.1420
    axis_b: float = 5.1420
    length_c: float = 28.1257
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

        return length_check & radial_check


@dataclass(frozen=True)
class Parallelepiped(Shape):
    """
    A 3D skewed box defined by three lattice vectors.
    """
    # Vectors defining the edges of the box (e.g. [10.0, 0.0, 0.0])
    vec_a: np.ndarray = field(default_factory=lambda: np.array([1, 0, 0]))
    vec_b: np.ndarray = field(default_factory=lambda: np.array([0, 1, 0]))
    vec_c: np.ndarray = field(default_factory=lambda: np.array([0, 0, 2]))

    center: np.ndarray = field(default_factory=lambda: np.array([0, 0, 0]))

    # Hidden field to store the pre-calculated inverse matrix
    # We use object.__setattr__ in __post_init__ because the class is frozen
    _inv_matrix: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        """
        Pre-calculate the matrix inverse so we don't do it 1 million times loop.
        Matrix M = [a, b, c] (columns)
        """
        # 1. Stack vectors as columns to make the Transformation Matrix
        matrix = np.column_stack((self.vec_a, self.vec_b, self.vec_c))

        # 2. Calculate Inverse (Cartesian -> Fractional)
        try:
            inv = np.linalg.inv(matrix)
        except np.linalg.LinAlgError:
            print("Warning: Parallelpiped vectors are coplanar (volume is zero). Using Identity.")
            inv = np.eye(3)

        # 3. Store it safely (bypassing frozen check)
        object.__setattr__(self, "_inv_matrix", inv)

    def is_inside(self, x, y, z) -> bool:
        # Shift to center
        dx = x - self.center[0]
        dy = y - self.center[1]
        dz = z - self.center[2]

        # Prepare for Matrix Multiplication
        # We need points as a (3, N) array: [[x1, x2...], [y1, y2...], [z1, z2...]]
        # x, y, z coming in are likely (Nx, Ny, Nz) grids. We flatten them for the dot product.

        # Flattening allows efficient batch processing
        points_flat = np.stack((dx.flatten(), dy.flatten(), dz.flatten()))

        # 3. Convert to Fractional Coordinates (u, v, w)
        # frac = M^-1 * r
        frac_coords = self._inv_matrix @ points_flat

        # 4. Check Bounds
        # If centered, valid range is [-0.5, 0.5]
        # We use absolute value to check both sides at once: |u| <= 0.5
        inside_flat = np.all(np.abs(frac_coords) <= 0.5, axis=0)

        # 5. Reshape back to original grid shape
        return inside_flat.reshape(x.shape)
