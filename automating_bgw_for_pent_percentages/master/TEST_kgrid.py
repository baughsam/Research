import pandas as pd


def generate_kgrid_input(csv_path, template_path, output_path, header_vectors):
    """
    Generates a 'kgrid.inp' file by updating a template with new coordinates
    and user-defined header vectors.

    Args:
        csv_path (str): The file path for the source CSV coordinates.
        template_path (str): The file path for the input template.
        output_path (str): The file path for the final output.
        header_vectors (list): A list of 3 lists, each containing 3 numbers for the new header.
    """
    print(f"Reading coordinates from '{csv_path}'...")
    # Read only the necessary columns from the provided CSV file
    df = pd.read_csv(csv_path)
    new_coords = df[['X_T_dep', 'Y_T_dep', 'Z_T_dep']].values.tolist()

    print(f"Reading template from '{template_path}'...")
    # Read all lines from the template file
    with open(template_path, 'r') as f:
        template_lines = f.readlines()

    # This list will hold the lines for the new file
    output_content = []

    '''
    # --- Step 1: Replace the header ---
    print("Replacing the first three lines with custom vectors...")
    for vector in header_vectors:
        # Format the new header lines with consistent spacing and precision
        line = f"   {vector[0]:.10f}   {vector[1]:.10f}   {vector[2]:.10f}\n"
        output_content.append(line)
    '''
    for i, vector in enumerate(header_vectors):
        # The first line (index 0) will be formatted as integers
        if i == 0:
            line = f"   {int(vector[0])}   {int(vector[1])}   {int(vector[2])}\n"
        # All other lines will be formatted as floats without extra padding
        else:
            line = f"   {vector[0]}   {vector[1]}   {vector[2]}\n"
        output_content.append(line)


    # --- Step 2: Add the rest of the file and replace coordinates ---
    # We assume the line with the number of atoms is the 4th line (index 3)
    # and the coordinates start on the 5th line.
    coord_start_index_in_template = 9

    # Add the line that specifies the number of atoms
    for j in range(3,9):
        output_content.append(template_lines[j])

    print(f"Replacing {len(new_coords)} atomic coordinates...")
    for i in range(len(new_coords)):
        # Get the original line from the template to extract the leading integer
        original_line = template_lines[coord_start_index_in_template + i]
        parts = original_line.strip().split()
        print(parts)
        leading_integer = parts[0]

        # Get the new coordinate values for this atom
        x, y, z = new_coords[i]

        # Format the new line, preserving the leading integer
        new_line = f"    {leading_integer}     {x:.10f}   {y:.10f}   {z:.10f}\n"
        output_content.append(new_line)
    ##addd in last lines
    for j in range(81,85):
        output_content.append(template_lines[j])

    # --- Step 3: Write the output file ---
    with open(output_path, 'w') as f:
        f.writelines(output_content)

    print(f"✅ Successfully created '{output_path}'.")


# --- Main execution block ---
if __name__ == "__main__":
    # ===================================================================
    # ==> EDIT THESE NINE NUMBERS for the first three lines.
    # Each inner list represents one line in the output file.
    # ===================================================================
    my_header_vectors = [
        [12.0, 0.0, 0.0],
        [0.0, 12.0, 0.0],
        [0.0, 0.0, 12.0]
    ]
    # ===================================================================

    # Define the input and output filenames
    CSV_FILE = '91cm-1_struc_1.csv'
    TEMPLATE_FILE = 'kgrid_template.inp'
    OUTPUT_FILE = ('kgrid_test.inp')

    # Run the main function
    generate_kgrid_input(
        csv_path=CSV_FILE,
        template_path=TEMPLATE_FILE,
        output_path=OUTPUT_FILE,
        header_vectors=my_header_vectors
    )