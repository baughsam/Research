import numpy as np
from dataclasses import dataclass, field
from ct_character.Shape import Shape



@dataclass(frozen=True)
class Configuration():
    lattice_vectors: np.ndarray         # 3x3 Matrix (Angstroms/Bohr)
    origin: np.ndarray                  # Vector (x,y,z)
    grid_shape: tuple[int, int, int]    # (Nx, Ny, Nz) e.g. (300,300,50)

    # Physics Model
    shape: Shape          # Shape Object from Shape.py

    # Atomic Structure
    atom_positions: np.ndarray # (N_atoms, 3)
    atom_types: np.ndarray     # (N_atoms, ) ~ will be integers (Atomic Numbers)



    @property
    def total_volume(self) -> float:
        """Calculates scalar triple product: a dot (b x c)"""
        v1, v2, v3 = self.lattice_vectors
        return np.abs(np.dot(v1, np.cross(v2,v3)))

    @property
    def dv(self) -> float:
        """Volume of a single voxel"""
        n_points = self.grid_shape[0] * self.grid_shape[1] * self.grid_shape[2]
        return self.total_volume / n_points

    @property
    def transform_matrix(self) -> np.ndarray:
        """
        Matrix to  convert Grid Indices (i,j,k) -> Cartesian (x,y,z)
        Equivalent to 'latticetoxyztransfermatrix'[cite: 62].
        """
        nx, ny, nz = self.grid_shape
        a, b, c = self.lattice_vectors

        step_a = a/nx
        step_b = b/ny
        step_c = c/nz

        return np.array([step_a, step_b, step_c])

    # Checks for bad data:
    def __post_init__(self):
        """
        Automatic Validation: Runs immediately after object creation.
        If IOHandler passes bad data, this crashes the program intentionally.
        """
        # CHECK 1: Lattice Shape
        if self.lattice_vectors.shape != (3, 3):
            raise ValueError(f"Lattice vectors must be 3x3 matrix. Got {self.lattice_vectors.shape}")

        # CHECK 2: Grid Constraints
        if any(x <= 0 for x in self.grid_shape):
            raise ValueError(f"Grid dimensions must be positive. Got {self.grid_shape}")

        # CHECK 3: Atom Consistency
        # Ensure we have the same number of positions as types
        if len(self.atom_positions) != len(self.atom_types):
            raise ValueError(f"Mismatch: {len(self.atom_positions)} positions but {len(self.atom_types)} types.")

        # CHECK 4: Data Types (Optional but safe)
        if not isinstance(self.shape, Shape):
            raise TypeError("The 'shape' field must be a valid Shape object (Cylinder/Ellipsoid).")