# process_qe_input.py

import sys
import os
from ase.io import read, write
from ase.visualize import view

def process_qe_input(file_path):
    """
    Reads a Quantum ESPRESSO input file, visualizes the atomic structure,
    and saves a copy in a VESTA-compatible format (.xsf).

    Args:
        file_path (str): The path to the Quantum ESPRESSO input file (.in or .pwi).
    """
    try:
        # Read the atomic structure from the Quantum ESPRESSO input file.
        atoms = read(file_path, format='espresso-in')
        print(f"Successfully read structure from '{file_path}'.")
        print("Number of atoms:", len(atoms))

        # --- New Section: Export for VESTA ---
        # Generate an output filename by changing the extension to .xsf
        base_name = os.path.splitext(file_path)[0]
        output_file = f"{base_name}.xsf"

        # Write the Atoms object to the .xsf file format
        # VESTA reads this format perfectly, including the unit cell.
        write(output_file, atoms)
        print(f"Structure saved to '{output_file}' for use in VESTA. ✅")
        # ------------------------------------

        # Launch the ASE GUI viewer to display the structure.
        print("\nOpening ASE's visualization window...")
        view(atoms, viewer='ase')

    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")
        print("\nCould not parse the structure. Please ensure the input file contains")
        print("valid 'CELL_PARAMETERS' and 'ATOMIC_POSITIONS' cards.")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    else:
        input_file = 'si.scf.in'
        print(f"No file path provided. Trying the default file: '{input_file}'")

    process_qe_input('111_input_3_i_orig.in')