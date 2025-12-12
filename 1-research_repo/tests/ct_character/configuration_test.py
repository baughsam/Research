from dataclasses import FrozenInstanceError
import numpy as np
import pytest
from ct_character.Exciton import Configuration
from ct_character.Shape import EllipticalCylinder

# Test Setup
@pytest.fixture
def dummy_shape():
    """Returns valid Shape object to use in tests."""
    return EllipticalCylinder(axis_a=10, axis_b=10, length_c=10)

@pytest.fixture
def orthogonal_config(dummy_shape):
    """Creates a 10x10x10 cubic system."""
    lattice = np.array([[10.0, 0.0,0.0],
                        [0.0, 10.0, 0.0],
                        [0.0, 0.0, 10.0]])
    grid = (10, 10, 10)
    # 1 atom at origin
    atoms_pos = np.zeros((1,3))
    atoms_type = np.array([1])

    return Configuration(
        lattice_vectors=lattice,
        origin=np.zeros(3),
        grid_shape = grid,
        shape = dummy_shape,
        atom_positions= atoms_pos,
        atom_types = atoms_type
    )

# Property methods tests
def test_volume_calculation(orthogonal_config):
    """Test 1: Correst volume size calculation."""
    # 10 * 10 * 10 = 1000.0
    assert np.isclose(orthogonal_config.total_volume, 1000.0)

def test_voxel_calculation(orthogonal_config):
    """Test 2: Correct voxel size calculation."""
    assert np.isclose(orthogonal_config.dv, 1)

def test_transform_matrix(orthogonal_config):
    """Test 3: Correct transformation matrix."""
    expected_matrix = np.array([[1.0, 0.0, 0.0],
                                [0.0, 1.0, 0.0],
                                [0.0, 0.0, 1.0]])
    assert np.allclose(orthogonal_config.transform_matrix, expected_matrix)

# Safety Tests

#Orthogonality ~ Non-Cubic Structures are Common, we need to make sure they work for our code
def test_orthoganality(dummy_shape):
    """
    SAFETY: Test a tilted (monoclinic) cell.
    Simple volume = L*W*H fails here, so this proves that our cross-product math works.
    """
    # A box tilted 45 degrees in X-Y plane
    lattice_non_orth = np.array([
                        [10.0, 0.0, 0.0],   # Vector A (Length 10)
                        [10.0, 10.0, 0.0],  # Vector B (Length 14.1, Height 10)
                        [0.0, 0.0, 10.0]])  # Vector C (Height 10)

    # Volume should be Base * Height * Depth = 10 * 10 * 10 = 1000
    # (The tilt shouldn't change the volume of this specific shape)

    config = Configuration(lattice_non_orth, np.zeros(3), (10, 10, 10),
                           dummy_shape, np.zeros((1, 3)), np.array([1]))

    assert np.isclose(config.total_volume, 1000.0)

def test_immutability(orthogonal_config):
    """
     SAFETY: Ensure Solver cannot accidentally change the lattice.
    """

    with pytest.raises(FrozenInstanceError):
        orthogonal_config.lattice_vectors = np.random.rand(3) * 50.0

def test_grid_safety(dummy_shape):
    """
    Ensure grid dimensions are physically possible
    """
    lattice = np.eye(3)
    bad_grid = (10, -5, 10)

    with pytest.raises(ValueError, match="positive"):
        Configuration(lattice, np.zeros(3), bad_grid,
                      dummy_shape, np.zeros((1, 3)), np.array([1]))