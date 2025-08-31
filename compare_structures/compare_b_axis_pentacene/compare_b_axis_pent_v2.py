import numpy as np
import sys
import csv

# ========================== USER CONFIGURATION ==========================

# 1. Specify the names of the two input files to compare.
#    IMPORTANT: Both files must be in Cartesian coordinates (in Angstroms).
FILE_1 = 'pristine_atom_pos_cart_angs.txt'
FILE_2 = '90.176552_cm-1_atom_pos_cart_angs.txt'

# 2. Specify the name for the output CSV file.
OUTPUT_CSV_FILENAME = 'discard.csv'

# 3. Define your lattice matrix (T) in Angstroms.
#    The b-axis is defined by the second column of this matrix.
#    T = [[a_x, b_x, c_x],
#         [a_y, b_y, c_y],
#         [a_z, b_z, c_z]]
LATTICE_MATRIX = np.array([
    [6.266, 0.7203432, 0.587676],
    [0.0, 7.741558672, 3.358122],
    [0.0, 0.0, 14.1244]
])


# ======================== END OF CONFIGURATION ========================


def read_coordinate_file(filename):
    """Reads an XYZ-like coordinate file and returns a list of atoms."""
    try:
        with open(filename, 'r') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"Error: Input file not found at '{filename}'")
        sys.exit(1)

    atoms = []
    for i, line in enumerate(lines):
        parts = line.strip().split()
        if not line.strip() or line.strip().startswith('#'):
            continue

        if len(parts) == 4:
            symbol = parts[0]
            try:
                coords = np.array([float(p) for p in parts[1:]])
                atoms.append({'symbol': symbol, 'pos': coords})
            except ValueError:
                print(f"Warning: Could not parse coordinates on line {i + 1} in {filename}. Skipping.")
        else:
            print(f"Warning: Malformed line {i + 1} in {filename}. Skipping.")

    return atoms


def compare_and_save_positions(file1, file2, lattice_matrix, csv_filename):
    """
    Compares two coordinate files, calculates displacement along the b-axis,
    prints the results, and saves them to a CSV file.
    """
    # --- 1. Get the b-axis unit vector ---
    b_vector = lattice_matrix[:, 1]
    norm_b = np.linalg.norm(b_vector)
    print(b_vector)

    if norm_b == 0:
        print("Error: The b-vector has zero length.")
        return

    b_unit_vector = b_vector / norm_b
    print(f"Analyzing displacement along the b-axis: {b_unit_vector}")
    print("-" * 50)

    # --- 2. Read atomic positions ---
    atoms1 = read_coordinate_file(file1)
    atoms2 = read_coordinate_file(file2)

    if len(atoms1) != len(atoms2):
        print("Warning: Files have different numbers of atoms. Comparing up to the smaller count.")
        min_atoms = min(len(atoms1), len(atoms2))
        atoms1, atoms2 = atoms1[:min_atoms], atoms2[:min_atoms]

    # --- 3. Open CSV file for writing ---
    with open(csv_filename, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)

        # Write CSV header
        csv_writer.writerow(['Atom_Index', 'Symbol', 'Displacement_on_b_axis_A'])

        # --- 4. Calculate, print, and save the displacement for each atom ---
        print(f"{'Atom':<8} {'Symbol':<8} {'Displacement on b-axis (Å)':>30}")
        print(f"{'=' * 8} {'=' * 8} {'=' * 30}")

        total_displacement = 0.0
        for i in range(len(atoms1)):
            pos1, pos2 = atoms1[i]['pos'], atoms2[i]['pos']
            symbol = atoms1[i]['symbol']

            diff_vector = pos2 - pos1
            displacement_along_b = np.dot(diff_vector, b_unit_vector)
            total_displacement += displacement_along_b

            # Print to console
            print(f"{i + 1:<8} {symbol:<8} {displacement_along_b:>30.8f}")

            # Write row to CSV
            csv_writer.writerow([i + 1, symbol, displacement_along_b])

    avg_displacement = total_displacement / len(atoms1) if atoms1 else 0.0
    print("-" * 50)
    print(f"Average displacement along the b-axis: {avg_displacement:.8f} Å")
    print(f"\nAnalysis complete! Results have been saved to '{csv_filename}' 💾")


if __name__ == '__main__':
    compare_and_save_positions(FILE_1, FILE_2, LATTICE_MATRIX, OUTPUT_CSV_FILENAME)