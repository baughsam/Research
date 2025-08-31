# Finds highest and lowest values in an eigenvalue_bx_noeh.dat

import sys


def find_min_max_ec(file_path):
    """
    Parses a data file to find the minimum and maximum values in the 'ec (eV)' column.

    Args:
        file_path (str): The path to the input data file.
    """
    # Initialize variables to store the min and max values.
    # We start with min as positive infinity and max as negative infinity
    # so that the first value read will always be lower/higher.
    lowest_ec = float('inf')
    highest_ec = float('-inf')

    line_count = 0

    try:
        with open(file_path, 'r') as f:
            for line in f:
                line_count += 1
                # Strip leading/trailing whitespace from the line
                line = line.strip()

                # Skip header lines. Based on your image, data starts after the line
                # that begins with '#'. We can make this robust.
                if not line or line.startswith('#') or line.startswith('nspin'):
                    continue

                # Split the line into columns based on whitespace
                columns = line.split()

                # The 'ec (eV)' value is in the 4th column (index 3)
                # Ensure the line has enough columns to avoid errors
                if len(columns) >= 7:
                    try:
                        # Convert the value to a float
                        ec_value = float(columns[6]) #Change this depending on which column

                        # Update the lowest and highest values
                        if ec_value < lowest_ec:
                            lowest_ec = ec_value

                        if ec_value > highest_ec:
                            highest_ec = ec_value

                    except ValueError:
                        # This handles cases where a column might not be a valid number
                        print(f"Warning: Could not parse number on line {line_count}: '{columns[3]}'")
                        continue

        # Check if any data was actually found
        if lowest_ec == float('inf'):
            print("No valid data was found in the 'ec (eV)' column.")
        else:
            print("Analysis complete. ✅")
            print(f"  Lowest: {lowest_ec}")
            print(f"  Highest: {highest_ec}")

    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    # Check if a file path is provided as a command-line argument.
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    else:
        # If no file is provided, prompt the user for it.
        input_file = "eigenvalues_b1_noeh_91cm-1.dat"

    if input_file:
        find_min_max_ec(input_file)
    else:
        print("No file provided. Exiting.")