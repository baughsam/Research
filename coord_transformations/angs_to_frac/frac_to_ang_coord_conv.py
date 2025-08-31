import numpy as np

# ========================== USER CONFIGURATION ==========================

# 1. Specify the name of your input file containing fractional coordinates.
#    The expected format is: Atom_Symbol Frac_X Frac_Y Frac_Z
INPUT_FILENAME = 'pristine_atom_pos_frac.txt'

# 2. Specify the name for the output file for the Cartesian coordinates.
OUTPUT_FILENAME = 'pristine_atom_pos_cart_angs.txt'

# 3. IMPORTANT: Define your lattice matrix (T) in Angstroms.
#    This must be the same matrix used to generate the fractional coordinates.
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


def convert_fractional_to_cartesian(in_file, out_file, lattice_matrix):
    """
    Reads atomic positions in fractional coordinates from a file, converts
    them to Cartesian coordinates, and writes them to another file.

    Args:
        in_file (str): Path to the input file (Fractional).
        out_file (str): Path to the output file (Cartesian).
        lattice_matrix (np.ndarray): 3x3 matrix where columns are the
                                     lattice vectors a, b, and c.
    """
    print(f"Reading fractional coordinates from '{in_file}'...")

    # Read all lines from the input file
    with open(in_file, 'r') as f_in:
        lines = f_in.readlines()

    # Open the output file to write the results
    with open(out_file, 'w') as f_out:
        f_out.write("# Atom_Symbol      Cart_X             Cart_Y             Cart_Z\n")

        # Process each line
        for i, line in enumerate(lines):
            # Strip whitespace and split the line by spaces
            parts = line.strip().split()

            # Skip comments and ensure the line has the correct format
            if not line.startswith('#') and len(parts) == 4:
                atom_symbol = parts[0]
                try:
                    # Create a numpy array of the fractional coordinates
                    frac_coords = np.array([float(coord) for coord in parts[1:]])

                    # Perform the matrix multiplication to get Cartesian coordinates
                    # c = T * f
                    cart_coords = np.matmul(lattice_matrix, frac_coords)

                    # Write the formatted output
                    output_line = (
                        f"{atom_symbol:<4} "
                        f"{cart_coords[0]:>18.12f} "
                        f"{cart_coords[1]:>18.12f} "
                        f"{cart_coords[2]:>18.12f}\n"
                    )
                    f_out.write(output_line)

                except ValueError:
                    print(f"Warning: Could not parse coordinates on line {i + 1}. Skipping.")
            elif line.strip() and not line.startswith('#'):
                print(f"Warning: Malformed line {i + 1} found. Skipping.")

    print(f"\nConversion complete! ✨")
    print(f"Cartesian coordinates have been written to '{out_file}'.")


if __name__ == '__main__':
    convert_fractional_to_cartesian(INPUT_FILENAME, OUTPUT_FILENAME, LATTICE_MATRIX)