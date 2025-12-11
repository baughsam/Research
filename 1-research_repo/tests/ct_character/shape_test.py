# test_shape.py
import pytest
import numpy as np
from ct_character.Shape import EllipticalCylinder


def test_cylinder_logic():
    # 1. Setup: Create a cylinder with radius=10, length=20, centered at (0,0,0)
    # Logic in Fortran Code: (x/a)^2 + (y/b)^2 <= 1 AND |z| <= c/2
    shape = EllipticalCylinder(axis_a=10, axis_b=10, length_c=20, center=np.array([0, 0, 0]))

    # 2. Test: The Center Point (Should be INSIDE)
    assert shape.is_inside(0, 0, 0) == True

    # 3. Test: A Point Way Outside (Should be OUTSIDE)
    assert shape.is_inside(100, 100, 100) == False

    # 4. Test: The Boundary (Exactly at Radius 10)
    # Floating point math can be tricky, so we test slightly inside/outside
    assert shape.is_inside(9.9, 0, 0) == True
    assert shape.is_inside(10.1, 0, 0) == False

    # 5. Test: The Length Boundary (z = 10 is edge, since length=20)
    assert shape.is_inside(0, 0, 9.9) == True
    assert shape.is_inside(0, 0, 10.1) == False