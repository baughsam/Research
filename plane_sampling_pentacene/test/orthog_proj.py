import numpy as np
import pandas as pd
import os


# --- Function from Previous Step ---
def find_parallel_planes(points, distance):
    if points.shape[0] < 3:
        raise ValueError("At least 3 points are required to define a plane.")
    centroid = points.mean(axis=0)
    centered_points = points - centroid
    _, _, vh = np.linalg.svd(centered_points)
    normal_vector = vh[-1, :]
    normal_vector = normal_vector / np.linalg.norm(normal_vector)
    d = -np.dot(normal_vector, centroid)
    return {
        "best_fit_plane": {"normal": normal_vector, "d": d},
        "plane_above": {"normal": normal_vector, "d": d - distance},
        "plane_below": {"normal": normal_vector, "d": d + distance},
    }


# --- Helper Function for Projection ---
def project_point_onto_plane(point, plane):
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
    above_filename = f"./csv_files/{base_filename}_projected_above.csv"
    df_above.to_csv(above_filename, index=False)
    print(f"Successfully saved projected 'above' coordinates to '{above_filename}'")

    # Convert the list for the 'below' plane to a DataFrame and save it
    df_below = pd.DataFrame(projected_below_coords, columns=headers)
    below_filename = f"./csv_files/{base_filename}_projected_below.csv"
    df_below.to_csv(below_filename, index=False)
    print(f"Successfully saved projected 'below' coordinates to '{below_filename}'")


# --- USAGE EXAMPLE ---
if __name__ == "__main__":
    # 1. Define the atoms to be used for creating the plane
    # For this example, we'll create some sample coordinates
    # In your real use case, you would load these from your CSV
    atoms_for_plane = np.array([
        [1.0, 1.1, 5.05], [3.0, 0.9, 5.01], [0.9, 3.1, 4.95], [3.1, 3.0, 4.99]
    ])

    # 2. Define all the atoms that you want to project onto the planes
    # This could be the same set of atoms or a different one
    all_atoms_to_project = np.array([
        [1.0, 1.1, 5.05], [3.0, 0.9, 5.01], [0.9, 3.1, 4.95],
        [3.1, 3.0, 4.99], [5.5, 2.0, 8.30], [2.0, 2.0, 0.50]
    ])

    # 3. Set the desired separation distance in Angstroms
    separation_distance = 1.5

    # 4. Calculate the best-fit and parallel planes
    print("Step 1: Calculating plane equations...")
    the_planes = find_parallel_planes(atoms_for_plane, separation_distance)
    print("Done.\n")

    # 5. Run the projection and save the results
    print("Step 2: Projecting coordinates and saving to CSV...")
    project_and_save_coordinates(the_planes, all_atoms_to_project, base_filename="final_atom_positions")
    print("Done.\n")

    # Verify that the files were created
    print("Generated files:")
    print(os.listdir())