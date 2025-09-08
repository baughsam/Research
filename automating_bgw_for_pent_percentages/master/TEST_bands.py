import pandas as pd
import sys


def create_bands_input(template_path, csv_path, kgrid_path, output_path, new_nbnd):
    """
    Generates a Quantum ESPRESSO input file for a 'bands' calculation.

    This script modifies a template file by:
    1. Updating the number of bands (nbnd).
    2. Replacing atomic coordinates with values from a CSV file.
    3. Appending the full content of 'kgrid.out' to the end.
    """
    print("Starting file generation...")

    # --- Step 1: Read all source files ---
    try:
        print(f"Reading template from '{template_path}'...")
        with open(template_path, 'r') as f:
            template_lines = f.readlines()

        print(f"Reading k-points from '{kgrid_path}'...")
        with open(kgrid_path, 'r') as f:
            kgrid_content = f.read()

        print(f"Reading coordinates from '{csv_path}'...")
        coords_df = pd.read_csv(csv_path)
        required_cols = ['X_T_dep', 'Y_T_dep', 'Z_T_dep']
        if not all(col in coords_df.columns for col in required_cols):
            print(f"Error: CSV file must contain the columns: {required_cols}")
            sys.exit(1)
        new_coords = coords_df[required_cols].values
        num_atoms = len(coords_df)  # Number of atoms is determined by CSV rows

    except FileNotFoundError as e:
        print(f"Error: File not found - {e.filename}")
        sys.exit(1)

    # --- Step 2: Find the atomic positions section in the template ---
    try:
        atom_pos_header_index = next(i for i, line in enumerate(template_lines) if 'ATOMIC_POSITIONS' in line)
    except StopIteration:
        print("Error: Template file is missing the 'ATOMIC_POSITIONS' card.")
        sys.exit(1)

    # Extract the original atom markers (e.g., 'C', 'H') from the template
    atom_markers = [line.strip().split()[0] for line in
                    template_lines[atom_pos_header_index + 1: atom_pos_header_index + 1 + num_atoms]]

    # --- Step 3: Build the new file content ---
    output_content = []

    # Use a flag to know when to skip old coordinate lines
    skip_lines = False
    lines_to_skip = num_atoms

    for line in template_lines:
        # Replace the nbnd value
        if '   nbnd		        =' in line:
            output_content.append(f"   nbnd             = {new_nbnd}\n")
            continue

        # When we find the ATOMIC_POSITIONS header...
        if 'ATOMIC_POSITIONS' in line:
            output_content.append(line)  # Add the header
            print(f"Replacing {num_atoms} atomic positions...")
            # Add all the new coordinates from the CSV
            for i, marker in enumerate(atom_markers):
                x, y, z = new_coords[i]
                new_pos_line = f"   {marker:<2}   {x:11.8f}   {y:11.8f}   {z:11.8f}\n"
                output_content.append(new_pos_line)
            # Set flag to start skipping the old coordinate lines
            skip_lines = True
            continue

        # If the skip flag is on, skip the correct number of lines
        if skip_lines and lines_to_skip > 0:
            lines_to_skip -= 1
            continue

        # Otherwise, just copy the line from the template
        output_content.append(line)

    # Finally, append the entire content of kgrid.out at the end
    print(f"Appending content from '{kgrid_path}'...")
    output_content.append("\n" + kgrid_content)

    # --- Step 4: Write the new file ---
    with open(output_path, 'w') as f:
        f.writelines(output_content)

    print(f"✅ Successfully created '{output_path}'.")


# ===================================================================
# --- MAIN EXECUTION BLOCK ---
# You can change the values here
# ===================================================================
if __name__ == "__main__":
    # ==> 1. Set the desired number of bands (nbnd)
    NUMBER_OF_BANDS = 1000

    # ==> 2. Define the input and output filenames
    TEMPLATE_FILE = 'bands_template'
    COORDINATES_CSV = '91cm-1_struc_2.csv'
    KGRID_FILE = 'kgrid.out'
    OUTPUT_FILE = 'bands.in'

    # ==> 3. Run the main function
    create_bands_input(
        template_path=TEMPLATE_FILE,
        csv_path=COORDINATES_CSV,
        kgrid_path=KGRID_FILE,
        output_path=OUTPUT_FILE,
        new_nbnd=NUMBER_OF_BANDS
    )