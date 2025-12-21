import pytest
import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from ct_character.Solver import Solver

### --- Mocks (Helper object for testing) --- ###

@dataclass
class MockShape:
    """
    A mock shape that lets us control what s 'inside'
    """
    def is_inside(self, X, Y, Z):
        # Default: Everything is inside
        return np.ones_like(X, dtype=bool) # an array like X, save everything inside is True

@dataclass
class MockConfig:
    """
    A mock configuration w/ simple Identity matrices
    """
    # 10x10x10 grid
    grid_shape: tuple = (10, 10, 10)
    # Simple Identity Lattice (1 stap = 1 Bohr)
    lattice_vectors: np.ndarray = field(default_factory=lambda: np.eye(3))
    shape: MockShape = field(default_factory=MockShape)

    @property
    def transform_matrix(self):
        # Identity transform for simplicity
        return np.eye(3)

@dataclass
class MockExcitonData:
    """
    A mock data container
    """
    grid_data: np.ndarray
    density_inside_shape: Optional[np.ndarray] = None
    ct_ratio: Optional[float] = None
    dipole_moment: Optional[np.ndarray] = None
    quadrupole_moment: Optional[np.ndarray] = None
    avg_r: Optional[float] = None
    avg_a: Optional[float] = None
    avg_b: Optional[float] = None
    avg_c: Optional[float] = None
    rdf_values: Optional[np.ndarray] = None
    rdf_distance: Optional[np.ndarray] = None
    density_distance: Optional[np.ndarray] = None

### --- Fixture (Setup Logic) --- ###

@pytest.fixture
def clean_solver():
    """
    This function runs before every single test.
    It creates a fresh Solver w/ an empty 10x10x10 grid
    """
    # Create empty grid
    empty_density_grid = np.zeros((10,10,10))

    # Create Mocks
    mock_config = MockConfig()
    mock_data = MockExcitonData(grid_data=empty_density_grid)

    # Return the Solver instance
    return Solver(mock_data, mock_config)

## --- Tests --- ###

def test_generate_coordinates(clean_solver):
    """
    Test 1:
    Are  X, Y, Z, and R generated w/ correct shapes and centering?
    """
    X, Y, Z, R = clean_solver._generate_coordinates()
    # Check Shape
    assert X.shape == (10, 10, 10)

    # Check Centering
    # Index 5 should be 0.0 because di = i - (10/2.0) = 5 - 5 = 0.0 Bohr
    assert np.isclose(X[5,5,5], 0.0, atol=1e-9)
    assert np.isclose(Y[5,5,5], 0.0, atol=1e-9)
    assert np.isclose(Z[5,5,5], 0.0, atol=1e-9)
    assert np.isclose(R[5,5,5], 0.0, atol=1e-9)

    # Check a point away from center
    # Index 6 is 1 step away -> 1.0 Bohr
    assert np.isclose(X[6,5,5], 1.0, atol=1e-9)

def test_apply_mask(clean_solver):
    """
    Test 2:
    Does the mask correctly filter data?
    """
    # Set all data to 1.0
    clean_solver.data.grid_data = np.ones((10,10,10))

    # Create fake mask where only the center point is True
    fake_mask = np.zeros((10,10,10), dtype=bool)
    fake_mask[5,5,5] = True

    clean_solver._apply_mask_to_density(fake_mask)

    # Center should be 1.0, everything else should be 0.0
    assert np.isclose(clean_solver.data.density_inside_shape[5,5,5], 1.0, atol=1e-9)
    assert np.isclose(clean_solver.data.density_inside_shape[0,0,0], 0.0, atol=1e-9)

def test_calculate_ct_ratio(clean_solver):
    """
    Test 3 (Math test):
    Charge Transfer Ration Calculation
    """
    # Total density = 100.0 (1.0 in 100 spots)
    clean_solver.data.grid_data = np.zeros((10,10,10))
    clean_solver.data.grid_data[0:10,0:10,0] = 1.0

    # Inside density = 25.0 (1.0 in 25 spots)
    clean_solver.data.density_inside_shape = np.zeros((10, 10 , 10))
    clean_solver.data.density_inside_shape[0:5,0:5,0] = 1.0

    ratio = clean_solver._calculate_ct_ratio()
    assert np.isclose(ratio, 0.25, atol=1e-6)

def test_calculate_multipoles_dipole(clean_solver):
    """
    Test 4 (Physics test):
    Place a charge at x=2.0. Dipole X should be 2.0
    """
    # Setup Grid
    clean_solver.data.grid_data = np.zeros((10,10,10))

    # Place Charge at Index [7,5,5]
    # Center is index 5. Index 7 is + 2 steps away in the x direction.
    # Lattice step is 1.0, thus Position = +2.0 Bohr
    clean_solver.data.grid_data[7, 5, 5] = 10.0 # Arbitrary magnitude

    # 3. Apply Mask (Allow everything)
    mask = np.ones((10,10,10), dtype=bool)
    clean_solver._apply_mask_to_density(mask)

    # Generate Coords needed for multipoles
    X, Y, Z, R = clean_solver._generate_coordinates()

    # Calculate
    clean_solver._calculate_multipoles(X, Y, Z)

    # Dipole
    dipole = clean_solver.data.dipole_moment

    assert np.isclose(dipole[0], 2.0, atol=1e-6) # X
    assert np.isclose(dipole[1], 0.0, atol=1e-6) # Y
    assert np.isclose(dipole[2], 0.0, atol=1e-6) # Z