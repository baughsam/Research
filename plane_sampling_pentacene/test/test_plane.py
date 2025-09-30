import numpy as np


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
    d_above = d - distance
    d_below = d + distance

    return {
        "best_fit_plane": {"normal": normal_vector, "d": d},
        "plane_above": {"normal": normal_vector, "d": d_above},
        "plane_below": {"normal": normal_vector, "d": d_below},
        "centroid": centroid
    }


# --- USAGE EXAMPLE ---
if __name__ == "__main__":
    # Suppose these are the XYZ coordinates of all atoms in your unit cell
    all_atomic_coords = np.array([
        [1.0, 1.1, 5.05],  # Atom 0
        [3.0, 0.9, 5.01],  # Atom 1
        [0.9, 3.1, 4.95],  # Atom 2
        [3.1, 3.0, 4.99],  # Atom 3
        [5.5, 2.0, 8.30],  # Atom 4 (not part of the plane)
        [2.0, 2.0, 0.50]  # Atom 5 (not part of the plane)
    ])

    # 1. Specify the set of atoms you want to define the plane
    # Here, we choose atoms 0, 1, 2, and 3, which are roughly on a plane around z=5
    selected_atom_indices = [0, 1, 2, 3]
    selected_points = all_atomic_coords[selected_atom_indices]

    # 2. Specify how many Angstroms above and below you want the parallel planes
    separation_distance = 1.5  # in Angstroms

    # 3. Run the function
    plane_results = find_parallel_planes(selected_points, separation_distance)

    # 4. Print the results
    print("--- Best-Fit Plane ---")
    normal = plane_results['best_fit_plane']['normal']
    d_fit = plane_results['best_fit_plane']['d']
    print(f"Normal Vector (a, b, c): {np.round(normal, 4)}")
    print(f"Equation: {normal[0]:.4f}x + {normal[1]:.4f}y + {normal[2]:.4f}z + ({d_fit:.4f}) = 0\n")

    print(f"--- Plane {separation_distance} Å Above ---")
    d_above = plane_results['plane_above']['d']
    print(f"Equation: {normal[0]:.4f}x + {normal[1]:.4f}y + {normal[2]:.4f}z + ({d_above:.4f}) = 0\n")

    print(f"--- Plane {separation_distance} Å Below ---")
    d_below = plane_results['plane_below']['d']
    print(f"Equation: {normal[0]:.4f}x + {normal[1]:.4f}y + {normal[2]:.4f}z + ({d_below:.4f}) = 0\n")

    # To get coordinates on a plane, you can fix two variables (e.g., x and y)
    # and solve for the third (z). For the "plane_above":
    # z = (-a*x - b*y - d_above) / c
    x_sample, y_sample = 2.0, 2.0
    if abs(normal[2]) > 1e-6:  # Avoid division by zero if plane is vertical
        z_sample_above = (-normal[0] * x_sample - normal[1] * y_sample - d_above) / normal[2]
        print(f"Example point on the 'above' plane at (x={x_sample}, y={y_sample}): z = {z_sample_above:.4f}")