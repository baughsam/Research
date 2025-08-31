import numpy as np

# ========================== USER CONFIGURATION ==========================

# 1. Specify the name of your input file containing Cartesian coordinates.
#    The expected format is: Atom_Symbol X Y Z (e.g., "C 4.7032 3.0516 0.1823")
INPUT_FILENAME = '91cm-1_atom_pos_cart_angs.txt'

# 2. Specify the name for the output file for the fractional coordinates.
OUTPUT_FILENAME = 'fractional_coords.txt'

# 3. IMPORTANT: Define your lattice matrix (T) in Angstroms.
#    The columns of this matrix should be your lattice vectors a, b, and c.
#    T = [[a_x, b_x, c_x],
#         [a_y, b_y, c_y],
#         [a_z, b_z, c_z]]
#
#    Using the pentacene lattice from your example script as a placeholder:
LATTICE_MATRIX = np.array([
    [6.266, 0.7203432, 0.587676],
    [0.0, 7.741558672, 3.358122],
    [0.0, 0.0, 14.1244]
])


# ======================== END OF CONFIGURATION ========================


def convert_cartesian_to_fractional(in_file, out_file, lattice_matrix):
    """
    Reads atomic positions in Cartesian coordinates from a file, converts
    them to fractional coordinates, and writes them to another file.

    Args:
        in_file (str): Path to the input file (Cartesian).
        out_file (str): Path to the output file (Fractional).
        lattice_matrix (np.ndarray): 3x3 matrix where columns are the
                                     lattice vectors a, b, and c.
    """
    print("Calculating the inverse of the lattice matrix...")
    try:
        inv_lattice_matrix = np.linalg.inv(lattice_matrix)
        print("Inverse matrix calculated successfully.")
    except np.linalg.LinAlgError:
        print("Error: The lattice matrix is singular and cannot be inverted.")
        return

    print(f"Reading Cartesian coordinates from '{in_file}'...")

    # Read all lines from the input file
    with open(in_file, 'r') as f_in:
        lines = f_in.readlines()

    # Open the output file to write the results
    with open(out_file, 'w') as f_out:
        f_out.write("# Atom_Symbol   Frac_X        Frac_Y        Frac_Z\n")

        # Process each line
        for i, line in enumerate(lines):
            parts = line.strip().split()

            # Ensure the line has the correct format (Symbol, X, Y, Z)
            if len(parts) == 4:
                atom_symbol = parts[0]
                try:
                    # Create a numpy array of the cartesian coordinates
                    cart_coords = np.array([float(coord) for coord in parts[1:]])

                    # Perform the matrix multiplication to get fractional coordinates
                    # f = T⁻¹ * c
                    frac_coords = np.matmul(inv_lattice_matrix, cart_coords)

                    # Write the formatted output
                    output_line = (
                        f"{atom_symbol:<4} "
                        f"{frac_coords[0]:>12.8f} "
                        f"{frac_coords[1]:>12.8f} "
                        f"{frac_coords[2]:>12.8f}\n"
                    )
                    f_out.write(output_line)

                except ValueError:
                    print(f"Warning: Could not parse coordinates on line {i + 1}. Skipping.")
            elif line.strip():  # If the line is not empty but also not valid
                print(f"Warning: Malformed line {i + 1} found. Skipping.")

    print(f"\nConversion complete! ✨")
    print(f"Fractional coordinates have been written to '{out_file}'.")


if __name__ == '__main__':
    convert_cartesian_to_fractional(INPUT_FILENAME, OUTPUT_FILENAME, LATTICE_MATRIX)