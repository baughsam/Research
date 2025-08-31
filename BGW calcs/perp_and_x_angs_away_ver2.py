import numpy as np

def get_perpendicular_positions_periodic(
    fractional_position,
    distance_x,
    unit_cell_vectors
):
    """
    Calculates two atomic positions perpendicular to the original position
    vector within a periodic unit cell.

    The function takes an initial position in fractional coordinates and converts
    it to Cartesian coordinates. It then generates a displacement vector that
    is perpendicular to this Cartesian position vector and has a length equal
    to the specified distance_x. The final positions are returned in both
    fractional and Cartesian coordinates.

    Args:
        fractional_position (np.ndarray or list): A 3D vector of the initial
                                                  atomic position in fractional
                                                  coordinates (relative to unit
                                                  cell vectors).
        distance_x (float): The desired distance for the perpendicular
                            displacement in angstroms.
        unit_cell_vectors (np.ndarray or list of lists): A 3x3 matrix where
                                                         each row is a unit
                                                         cell vector in
                                                         angstroms.
                                                         Example: [[a1, a2, a3],
                                                                   [b1, b2, b3],
                                                                   [c1, c2, c3]]

    Returns:
        dict: A dictionary containing the two new positions in both 'fractional'
              and 'cartesian' coordinates. For example:
              {
                  'pos1_fractional': np.ndarray,
                  'pos1_cartesian': np.ndarray,
                  'pos2_fractional': np.ndarray,
                  'pos2_cartesian': np.ndarray,
              }
    """
    # Ensure inputs are NumPy arrays for vector/matrix operations
    frac_pos = np.array(fractional_position)
    lattice_matrix = np.array(unit_cell_vectors)

    # 1. Convert fractional coordinates to Cartesian coordinates
    # r_cart = f_a * a + f_b * b + f_c * c
    initial_cart_pos = np.dot(frac_pos, lattice_matrix)

    # 2. Find a perpendicular vector in Cartesian space (same logic as before)
    # To find a vector perpendicular to the initial position vector, we take
    # the cross product with a non-collinear vector.
    if np.allclose(initial_cart_pos[:2], [0, 0]):
        # Vector is on or near the z-axis, so use the x-axis for cross product
        non_collinear_vector = np.array([1.0, 0.0, 0.0])
    else:
        # Otherwise, the z-axis is a safe choice
        non_collinear_vector = np.array([0.0, 0.0, 1.0])

    perpendicular_vector = np.cross(initial_cart_pos, non_collinear_vector)

    # Normalize the perpendicular vector to get a unit vector
    norm_perp_vector = perpendicular_vector / np.linalg.norm(perpendicular_vector)

    # Scale by the desired distance to get the Cartesian displacement vector
    displacement_vector_cart = norm_perp_vector * distance_x

    # 3. Calculate the two new positions in Cartesian coordinates
    pos1_cart = initial_cart_pos + displacement_vector_cart
    pos2_cart = initial_cart_pos - displacement_vector_cart

    # 4. Convert the new Cartesian positions back to fractional coordinates
    inv_lattice_matrix = np.linalg.inv(lattice_matrix)
    pos1_frac = np.dot(pos1_cart, inv_lattice_matrix)
    pos2_frac = np.dot(pos2_cart, inv_lattice_matrix)

    return {
        'pos1_fractional': pos1_frac,
        'pos1_cartesian': pos1_cart,
        'pos2_fractional': pos2_frac,
        'pos2_cartesian': pos2_cart,
    }


# --- Example Usage ---

# 1. Define the unit cell vectors (e.g., a simple orthorhombic cell in Angstroms)
#    Here, a=5Å, b=6Å, c=7Å along the Cartesian axes.

cell_vectors = [
    [6.266, 0.0, 0.0],
    [0.7203432, 7.741558672, 0.0],
    [0.587676, 3.358122, 14.1244]
]

# 2. Define the initial atomic position in FRACTIONAL coordinates
#    This position is at the body center of the orthorhombic cell.
initial_frac_pos = [4.69220255, 3.80098575, 2.117331]

# 3. Set the desired perpendicular distance 'x' in Angstroms
perpendicular_distance = 0.5

# 4. Get the new positions
new_positions = get_perpendicular_positions_periodic(
    initial_frac_pos,
    perpendicular_distance,
    cell_vectors
)

# --- Print the results ---
print("INPUTS")
print(f"Unit Cell (Å):\n{np.array(cell_vectors)}")
print(f"Initial Fractional Position: {np.array(initial_frac_pos)}")
print(f"Desired Displacement (x): {perpendicular_distance} Å")
print("-" * 40)

print("RESULTS")
print(f"Position 1 (Fractional): {new_positions['pos1_fractional']}")
print(f"Position 1 (Cartesian Å):  {new_positions['pos1_cartesian']}")
print("-" * 40)
print(f"Position 2 (Fractional): {new_positions['pos2_fractional']}")
print(f"Position 2 (Cartesian Å):  {new_positions['pos2_cartesian']}")