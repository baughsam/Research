import numpy as np
from dataclasses import dataclass, field
from ct_character.Shape import Shape
from typing import Optional


@dataclass
class ExcitonData():
    #self.variable: type # Variable from fortran code
    # Raw Data
    grid_data: np.ndarray # twopartcorr_INPUT

    # Middle Work
    density_inside_shape: Optional[np.ndarray] = None # twopartcorr_Volume
    density_distance: Optional[np.ndarray] = None
    total_weight: float = 0.0

    # Final Results
    ct_ratio: Optional[float] = None          # INVOLUMEFRACTION
    dipole_moment: Optional[np.ndarray] = None       # Dipole #[x, y, z]
    quadrupole_moment: Optional[np.ndarray] = None   # Quadrupole # 3x3 Matrix

    # Plotting Parameters
    rdf_distance: Optional[np.ndarray] = None   # X-axis
    rdf_counts: Optional[np.ndarray] = None     # Y-axis

    # --- 4. LEGACY METRICS (Fortran Style) ---
    # These represent "Average Density per Voxel".
    # CRITICAL: Summing these DOES NOT give the probability integral (missing 4*pi*r^2).
    # Used to verify agreement with legacy Fortran code.
    rdf_density_total: Optional[np.ndarray] = None  # Normalized Density (Total)
    rdf_density_in_vol: Optional[np.ndarray] = None  # Normalized Density (In-Volume)
    avg_r_fortran: Optional[float] = None  # Typically underestimates <r> (e.g. ~0.48 vs 1.5)

    # --- 5. EXACT METRICS (Physics Style) ---
    # These represent "Probability Mass".
    # CRITICAL: Summing `rdf_probability_in_vol` gives the integral for Eq 9.
    # These include the Jacobian volume element implicitly.
    rdf_probability_total: Optional[np.ndarray] = None  # Probability Mass P(r) (Total)
    rdf_probability_in_vol: Optional[np.ndarray] = None  # Probability Mass P(r) (In-Volume)`

    # First Moments ,|x|>
    # Average distance values (to replicate _OUT.txt in .f90 file)
    avg_r: Optional[float] = None  # <|r|>
    avg_a: Optional[float] = None  # <|a|>
    avg_b: Optional[float] = None  # <|b|>
    avg_c: Optional[float] = None  # <|c|>

    # Second Moments <x^2>
    avg_r2: Optional[float] = None
    avg_a2: Optional[float] = None
    avg_b2: Optional[float] = None
    avg_c2: Optional[float] = None




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

        # Transpose matrix so vectors -> columns
        return np.array([step_a, step_b, step_c]).T

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