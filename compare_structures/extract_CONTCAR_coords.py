# main.py
import os
import csv
from pymatgen.core import Structure

def contcar_to_csv(contcar_content, output_csv_filename):
    """
    Reads atomic coordinates from VASP CONTCAR file content,
    and saves them to a CSV file.

    Args:
        contcar_content (str): A string containing the content of the CONTCAR file.
        output_csv_filename (str): The name of the output CSV file.
    """
    try:
        # Create a temporary file to write the CONTCAR content
        # pymatgen's Structure.from_file expects a file path.
        temp_contcar_file = "CONTCAR_temp"
        with open(temp_contcar_file, "w") as f:
            f.write(contcar_content)

        # Load the structure from the CONTCAR file
        # The from_file method can automatically detect the file format.
        structure = Structure.from_file(temp_contcar_file)

        # Clean up the temporary file
        os.remove(temp_contcar_file)

        # Open the desired output CSV file in write mode
        with open(output_csv_filename, 'w', newline='') as csvfile:
            # Create a csv writer object
            csv_writer = csv.writer(csvfile)

            # Write the header row to the CSV file
            csv_writer.writerow(['element', 'frac_a', 'frac_b', 'frac_c'])

            # Iterate through each atom (site) in the structure
            for site in structure:
                # Get the element symbol (e.g., 'Si', 'O')
                element = site.species_string

                # Get the Cartesian coordinates of the atom
                coords = site.frac_coords

                # Write the element symbol and its x, y, z coordinates to the CSV
                csv_writer.writerow([element, coords[0], coords[1], coords[2]])

        print(f"Successfully extracted coordinates to '{output_csv_filename}'")

    except Exception as e:
        print(f"An error occurred: {e}")
        # Clean up the temporary file in case of an error
        if os.path.exists(temp_contcar_file):
            os.remove(temp_contcar_file)

def read_contcar_file(filepath):
     with open(filepath, 'r') as f:
         return f.read()

my_contcar_path = "CONTCAR_QE_pent"
contcar_data = read_contcar_file(my_contcar_path)
contcar_to_csv(contcar_data, "struc_coord_QE_pent.csv")