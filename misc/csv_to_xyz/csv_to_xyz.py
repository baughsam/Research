import pandas as pd
import argparse


def create_xyz_from_csv(input_csv, output_xyz, atom_symbols, atom_counts):
    """
    Converts a CSV file with atomic coordinates (X, Y, Z) into an XYZ file.

    Args:
        input_csv (str): Path to the input CSV file.
        output_xyz (str): Path for the generated output XYZ file.
        atom_symbols (list): A list of atomic symbols (e.g., ['C', 'H']).
        atom_counts (list): A list of integers corresponding to the count of each atom.
    """
    # Step 1: Validate inputs
    if len(atom_symbols) != len(atom_counts):
        print("Error: The number of atom symbols must match the number of atom counts.")
        return

    total_atoms = sum(atom_counts)
    print(f"Expecting a total of {total_atoms} atoms.")

    # Step 2: Read the coordinate data from the CSV file
    try:
        df = pd.read_csv(input_csv)
        # Ensure the required columns exist
        if not {'X', 'Y', 'Z'}.issubset(df.columns):
            print(f"Error: Input CSV file '{input_csv}' must contain 'X', 'Y', and 'Z' columns.")
            return

        # Extract coordinates, ensuring we only take as many as we need
        coords = df[['X', 'Y', 'Z']].values
        if len(coords) < total_atoms:
            print(f"Error: The CSV file contains {len(coords)} rows, but {total_atoms} atoms were specified.")
            return

    except FileNotFoundError:
        print(f"Error: The file '{input_csv}' was not found.")
        return
    except Exception as e:
        print(f"An error occurred while reading the CSV file: {e}")
        return

    # Step 3: Create the list of atom labels in the correct order
    labels = []
    for symbol, count in zip(atom_symbols, atom_counts):
        labels.extend([symbol] * count)

    # Step 4: Write the data to the XYZ file
    try:
        with open(output_xyz, 'w') as f:
            # First line: total number of atoms
            f.write(f"{total_atoms}\n")
            # Second line: a comment (we'll use the input filename)
            f.write(f"Generated from {input_csv}\n")

            # Subsequent lines: AtomSymbol X Y Z
            for i in range(total_atoms):
                symbol = labels[i]
                x, y, z = coords[i]
                f.write(f"{symbol:2s} {x:15.8f} {y:15.8f} {z:15.8f}\n")

        print(f"Successfully created '{output_xyz}' with {total_atoms} atoms.")

    except Exception as e:
        print(f"An error occurred while writing the XYZ file: {e}")


if __name__ == "__main__":
    #input_csv = "/home/baughs/PycharmProjects/Research/frozen_phonons/ZG/Complete_ZG_code_pent/90.176552_cm-1_posvec+eigvec.csv"
    input_csv = "plane_sampling_test.csv"
    #output_xyz = "pentacene.xyz"
    output_xyz = "test_plane.xyz"
    atoms = ["C","H"]
    #counts = [44, 28]
    counts =[8,2]
    create_xyz_from_csv(input_csv, output_xyz, atoms, counts)
