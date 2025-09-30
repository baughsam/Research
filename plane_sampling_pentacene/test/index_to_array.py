import pandas as pd
import numpy as np


def get_atom_groups_from_csv(file_path, index_groups):
    """
    Reads a CSV file of atomic coordinates and extracts specific groups of atoms.

    Args:
        file_path (str): The path to the CSV file. The file should have columns
                         'Index', 'x_pos', 'y_pos', and 'z_pos'.
        index_groups (list of list of int): A list containing one or more lists
                                             of atom indices to extract.

    Returns:
        list of np.ndarray: A list where each element is a NumPy array of shape
                            (N, 3) containing the xyz coordinates for one of the
                            specified atom groups.
    """
    try:
        # Read the CSV, using the 'Index' column as the DataFrame index for easy selection
        df = pd.read_csv(file_path, index_col='Index')
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found. Please ensure it is uploaded and the path is correct.")
        return []
    except KeyError:
        print("Error: The CSV must contain an 'Index' column for this function to work.")
        return []

    coordinate_arrays = []
    # Loop through each list of indices provided by the user
    for group in index_groups:
        try:
            # Select rows based on the current group of indices (.loc)
            # and specify the columns we want
            selected_atoms = df.loc[group][['x_pos', 'y_pos', 'z_pos']]
            # Convert the resulting DataFrame to a NumPy array and add to our list
            coordinate_arrays.append(selected_atoms.to_numpy())
        except KeyError as e:
            # This handles cases where an index in a group doesn't exist in the file
            print(f"Warning: One or more indices in group {group} not found in the CSV file: {e}. Skipping this group.")

    return coordinate_arrays


# --- USAGE EXAMPLE ---
if __name__ == "__main__":
    csv_file = '../plane_sampling/plane_sampling_test.csv'

    # Define the groups of atom indices you want to extract.
    # In this example, we want two separate groups (and therefore two separate arrays).
    atom_indices_to_get = [
        [1, 2, 3, 4],  # First group of atoms for the first plane
        [5, 6, 7, 8]  # Second group of atoms for a second plane
    ]

    # Call the function to get the list of coordinate arrays
    list_of_coordinate_arrays = get_atom_groups_from_csv(csv_file, atom_indices_to_get)
    print(list_of_coordinate_arrays[0])

"""
    # Print the results for verification
    if list_of_coordinate_arrays:
        print(f"Successfully extracted {len(list_of_coordinate_arrays)} groups of atoms.\n")

        for i, arr in enumerate(list_of_coordinate_arrays):
            print(f"--- Array for Group {i + 1} (Indices: {atom_indices_to_get[i]}) ---")
            print(arr)
            print(f"Shape: {arr.shape}\n")
"""