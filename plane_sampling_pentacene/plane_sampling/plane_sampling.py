import numpy as np
import pandas as pd
import glob
import os


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

    return coordinate_arrays # a list of coordinate arrays


def find_parallel_planes(points, distance):
    """
    Finds the best-fit plane for a set of 3D points and defines two
    parallel planes at a specified distance above and below it.

    Args:
        points (np.ndarray): A NumPy array of shape (N, 3) where N is the
                             number of points.
        distance (float): The perpendicular distance for the parallel planes.

    Returns:
        dict: A dictionary containing the parameters for the best-fit plane,
              the plane above, and the plane below. Each plane is defined by
              its normal vector (a, b, c) and constant d from the equation
              ax + by + cz + d = 0.
    """
    if points.shape[0] < 3:
        raise ValueError("At least 3 points are required to define a plane.")

    # 1. Find the centroid of the points. The best-fit plane must pass through it.
    centroid = points.mean(axis=0)

    # 2. Center the points by subtracting the centroid.
    centered_points = points - centroid

    # 3. Perform Singular Value Decomposition (SVD).
    # The normal to the best-fit plane is the last singular vector,
    # which corresponds to the smallest singular value.
    # In NumPy's `svd`, this is the last row of the `vh` matrix.
    _, _, vh = np.linalg.svd(centered_points)
    normal_vector = vh[-1, :]

    # Ensure the normal vector is a unit vector (it should be from SVD, but good practice)
    normal_vector = normal_vector / np.linalg.norm(normal_vector)

    # 4. Define the plane equation: ax + by + cz + d = 0
    # We have the normal vector (a, b, c). We can find d by substituting the centroid.
    # a*x_c + b*y_c + c*z_c + d = 0 => d = -(a*x_c + b*y_c + c*z_c)
    a, b, c = normal_vector
    d = -np.dot(normal_vector, centroid)

    # 5. Define the parallel planes. They have the same normal vector but a different d.
    # The distance between ax+by+cz+d1=0 and ax+by+cz+d2=0 is |d1-d2|/sqrt(a^2+b^2+c^2).
    # Since our normal is a unit vector, the distance is just |d1-d2|.
    d_above = d + distance
    d_below = d - distance

    return {
        "best_fit_plane": {"normal": normal_vector, "d": d},
        "plane_above": {"normal": normal_vector, "d": d_above},
        "plane_below": {"normal": normal_vector, "d": d_below},
        "centroid": centroid
    }

# --- Helper Function for Projection ---
def project_point_onto_plane(point, plane):
    """
    Calculates the orthogonal projection of a point onto a plane.

    Args:
        point (np.ndarray): The (x, y, z) coordinates of the atom.
        plane (dict): A dictionary with 'normal' vector and 'd' constant.

    Returns:
        np.ndarray: The projected (x, y, z) coordinates on the plane.
    """
    normal = plane['normal']
    d = plane['d']
    s = np.dot(normal, point) + d
    projected_point = point - s * normal
    return projected_point


# --- NEW FUNCTION TO PROJECT AND SAVE ---
def project_and_save_coordinates(plane_results, atom_coords, base_filename="projection_results"):
    """
    Projects a set of atom coordinates onto parallel planes and saves to CSV.

    Args:
        plane_results (dict): The dictionary output from find_parallel_planes().
        atom_coords (np.ndarray): An array of atom coordinates to project.
        base_filename (str): The base name for the output CSV files.
    """
    # Extract the plane definitions from the results dictionary
    plane_above = plane_results['plane_above']
    plane_below = plane_results['plane_below']

    # Create lists to hold the new projected coordinates
    projected_above_coords = []
    projected_below_coords = []

    # Loop through each atom and project it onto both planes
    for atom in atom_coords:
        projected_above_coords.append(project_point_onto_plane(atom, plane_above))
        projected_below_coords.append(project_point_onto_plane(atom, plane_below))

    # Define the headers for the CSV file
    headers = ['x_proj', 'y_proj', 'z_proj']

    # Convert the list for the 'above' plane to a DataFrame and save it
    df_above = pd.DataFrame(projected_above_coords, columns=headers)
    above_filename = f"./proj_files_csv/{base_filename}_projected_above.csv"
    df_above.to_csv(above_filename, index=False)
    print(f"Successfully saved projected 'above' coordinates to '{above_filename}'")

    # Convert the list for the 'below' plane to a DataFrame and save it
    df_below = pd.DataFrame(projected_below_coords, columns=headers)
    below_filename = f"./proj_files_csv/{base_filename}_projected_below.csv"
    df_below.to_csv(below_filename, index=False)
    print(f"Successfully saved projected 'below' coordinates to '{below_filename}'")


def combine_csv_files(input_folder, output_file):
    """
    Combines all CSV files in a given folder into a single CSV file.

    Args:
        input_folder (str): The path to the folder containing the CSV files.
        output_file (str): The full path for the new, combined CSV file.
    """
    try:
        # Use glob to get a list of all csv files in the folder
        all_files = glob.glob(os.path.join(input_folder, "*.csv"))

        if not all_files:
            print(f"No CSV files found in the directory: {input_folder}")
            return

        # Create a list to hold the individual dataframes
        df_list = [pd.read_csv(file) for file in all_files]

        # Concatenate all the dataframes into a single one
        combined_df = pd.concat(df_list, ignore_index=True)

        # Save the combined dataframe to a new CSV file
        combined_df.to_csv(output_file, index=False)

        print(f"✅ Successfully combined {len(all_files)} files into {output_file}")

    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":

    # --- Number of Planes ---
    num_of_planes = 2 #perhaps put a check that stops things if this number isn't equal to the number of lists in the index list

    # --- Pentacene Plane ---
    index_list =[[35,33,31,37,39,29,27,41,44,23,24,26,43,42,28,30,40,38,32,34,36,25],[12,14,16,10,18,8,20,6,21,4,1,2,3,22,5,19,7,17,9,15,11,13]]
    plane_csv = 'pentacene_coords_angs.csv'

    # --- TEST PLANE ---
    #index_list = [[1,2,3,4],[5,6,7,8]] #TEST LIST
    #plane_csv = "plane_sampling_test.csv"

    # --- Distance Above Planes ---
    dist_above_plane_angs = 0.01

    coordinate_arrays = get_atom_groups_from_csv(plane_csv, index_list)  #grabbing the atoms at these indexes, separated by a visual observation of which would be in a plane

# --- Finds Parallel Planes for each Plane ---
    for i in range(num_of_planes):
        FPP = find_parallel_planes(coordinate_arrays[i], dist_above_plane_angs)
        project_and_save_coordinates(FPP, coordinate_arrays[i], base_filename=f"plane_{i+1}")


    # --- Combines Individual Projections into 1 .csv File ---
    #because of this, be sure to re-run so that you know the files in proj_files_csv are the correct ones
    combine_csv_files("./proj_files_csv/", "./single_proj_csv/single_proj.csv")