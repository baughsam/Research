import numpy as np
import pytest
from ct_character.Exciton import ExcitonData


def test_exciton_data_defaults():
    """Test 1: Correct default data"""

    # Setup: Create data with just a dummy density
    dummy_density = np.zeros((10, 10, 10))
    data = ExcitonData(grid_data=dummy_density)

    # Test: Inputs should be stored
    assert np.array_equal(data.density, dummy_density)

    # Test: Results should be empty (Safety Check)
    assert data.density_inside_shape is None
    assert data.density_distance is None
    assert data.ct_ratio is None
    assert data.dipole_moment is None
    assert data.quadrupole_moment is None
    assert data.rdf_distance is None
    assert data.rdf_values is None

def test_exciton_data_mutability():
    """Test 2: Mutability status of ExcitonData"""
    dummy_density = np.zeros((10, 10, 10))
    data = ExcitonData(grid_data=dummy_density)
    data.ct_ratio = 1

    assert data.ct_ratio == 1