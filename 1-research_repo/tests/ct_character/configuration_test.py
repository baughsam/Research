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

def test_volume_calculation(orthogonal_config):
    """Test 1: Correst volume size calculation."""
    # 10 * 10 * 10 = 1000.0
    assert np.isclose(orthogonal_config.total_volume, 1000.0)