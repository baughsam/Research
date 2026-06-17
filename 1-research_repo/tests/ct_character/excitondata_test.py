import numpy as np
import pytest
from ct_character.Exciton import ExcitonData

def test_exciton_data_defaults():
    """Test 1: Correct default data for refactored fields"""
    dummy_density = np.zeros((10, 10, 10))
    data = ExcitonData(grid_data=dummy_density)

    # Test: Core input stored
    assert np.array_equal(data.grid_data, dummy_density)

    # Test: New streamlined fields default to None/0.0
    assert data.total_weight == 0.0
    assert data.ct_ratio is None
    assert data.rdf_distance is None
    assert data.rdf_counts is None
    assert data.rdf_probability_total is None
    assert data.rdf_probability_in_vol is None

def test_exciton_data_mutability():
    """Test 2: Mutability status of ExcitonData"""
    dummy_density = np.zeros((10, 10, 10))
    data = ExcitonData(grid_data=dummy_density)
    data.ct_ratio = 1.0

    assert data.ct_ratio == 1.0