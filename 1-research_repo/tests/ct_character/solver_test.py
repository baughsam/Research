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
    Test if X, Y, Z, R are generated w/ correct shapes and centering
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