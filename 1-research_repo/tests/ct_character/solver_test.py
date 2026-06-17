import pytest
import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from ct_character.Solver import Solver


### --- Mocks --- ###

@dataclass
class MockShape:
    def is_inside(self, X, Y, Z):
        return np.ones_like(X, dtype=bool)  # Everything is inside the shape


@dataclass
class MockConfig:
    grid_shape: tuple = (10, 10, 10)
    lattice_vectors: np.ndarray = field(default_factory=lambda: np.eye(3))
    shape: MockShape = field(default_factory=MockShape)
    dv: float = 1.0

    @property
    def transform_matrix(self):
        return np.eye(3)  # Identity matrix for easy Cartesian mapping


@dataclass
class MockExcitonData:
    grid_data: np.ndarray
    total_weight: float = 0.0
    ct_ratio: Optional[float] = None


### --- Fixture --- ###

@pytest.fixture
def clean_solver():
    empty_density_grid = np.zeros((10, 10, 10))
    mock_config = MockConfig()
    mock_data = MockExcitonData(grid_data=empty_density_grid)
    return Solver(mock_data, mock_config, do_rdf_analysis=False)


### --- Tests --- ###

def test_precompute_xy_grid(clean_solver):
    """Test if the base XY slice grid is generated with correct shapes."""
    xy_base = clean_solver._precompute_xy_grid(10, 10)
    assert xy_base.shape == (3, 10, 10)


def test_generate_slice_coords(clean_solver):
    """Test slice coordinate tracking and centering limits."""
    xy_base = clean_solver._precompute_xy_grid(10, 10)
    z_step_vec = clean_solver.config.transform_matrix[:, 2]

    # Target the true origin slice (k=4 yields dk = 4 - (10/2 - 1) = 0.0)
    X, Y, Z, R = clean_solver._generate_slice_coords(k=4, nz=10, xy_base=xy_base, z_step_vec=z_step_vec, calc_r=True)

    assert X.shape == (10, 10)

    # According to the formula: coordinate = index - 4.0
    # Therefore, index [4, 4, 4] is the absolute coordinate origin (0,0,0)
    assert np.isclose(X[4, 4], 0.0, atol=1e-7)
    assert np.isclose(Y[4, 4], 0.0, atol=1e-7)
    assert np.isclose(Z[4, 4], 0.0, atol=1e-7)
    assert np.isclose(R[4, 4], 0.0, atol=1e-7)

    # Verify a point 1 step away from the origin (Index 5 -> 5 - 4 = 1.0 Bohr)
    assert np.isclose(X[5, 4], 1.0, atol=1e-7)


def test_accumulate_raw_sums(clean_solver):
    """Test that slice-by-slice accumulation maps properly to the accumulator dictionary."""
    acc = clean_solver._initialize_accumulators()
    rho_slice = np.ones((10, 10))
    rho_inside = np.ones((10, 10)) * 0.5

    clean_solver._accumulate_raw_sums(acc, rho_slice, rho_inside)

    assert acc['total_density'] == 100.0
    assert acc['masked_density'] == 50.0


def test_solve_integration_empty_grid(clean_solver):
    """Verify entire pipeline completes successfully and calculates correct baseline defaults."""
    clean_solver.solve()

    assert clean_solver.data.ct_ratio is not None
    assert np.isclose(clean_solver.data.ct_ratio, 0.0, atol=1e-6)
    assert np.isclose(clean_solver.data.total_weight, 0.0, atol=1e-6)


def test_solve_with_uniform_density(clean_solver):
    """Verify baseline math calculations when grid data is filled completely."""
    clean_solver.data.grid_data = np.ones((10, 10, 10))
    clean_solver.solve()

    # Since MockShape says everything is inside, CT ratio must be 1.0
    assert np.isclose(clean_solver.data.ct_ratio, 1.0, atol=1e-6)
    assert np.isclose(clean_solver.data.total_weight, 1000.0, atol=1e-6)  # 10x10x10 * dv(1.0)