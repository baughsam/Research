# If using a Jupyter Notebook, add this line to the top of the cell:
# %matplotlib widget

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def find_parallel_planes(points, distance):
    if points.shape[0] < 3:
        raise ValueError("At least 3 points are required to define a plane.")
    centroid = points.mean(axis=0)
    centered_points = points - centroid
    _, _, vh = np.linalg.svd(centered_points)
    normal_vector = vh[-1, :]
    normal_vector = normal_vector / np.linalg.norm(normal_vector)
    a, b, c = normal_vector
    d = -np.dot(normal_vector, centroid)
    d_above = d - distance
    d_below = d + distance

    return {
        "best_fit_plane": {"normal": normal_vector, "d": d},
        "plane_above": {"normal": normal_vector, "d": d_above},
        "plane_below": {"normal": normal_vector, "d": d_below},
        "centroid": centroid
    }


def plot_atoms_and_planes(all_atoms, selected_atoms, plane_results, distance):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(all_atoms[:, 0], all_atoms[:, 1], all_atoms[:, 2], c='blue', label='All Atoms', s=50, alpha=0.6)
    ax.scatter(selected_atoms[:, 0], selected_atoms[:, 1], selected_atoms[:, 2], c='red', label='Selected Atoms', s=150,
               edgecolors='k')

    min_x, max_x = selected_atoms[:, 0].min() - 1, selected_atoms[:, 0].max() + 1
    min_y, max_y = selected_atoms[:, 1].min() - 1, selected_atoms[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(min_x, max_x, 10), np.linspace(min_y, max_y, 10))

    planes_to_plot = {
        'best_fit_plane': ('green', 0.2, 'Best-Fit Plane'),
        'plane_above': ('purple', 0.15, f'Plane +{distance}Å'),
        'plane_below': ('orange', 0.15, f'Plane -{distance}Å'),
    }

    for name, (color, alpha, label) in planes_to_plot.items():
        normal = plane_results[name]['normal']
        d = plane_results[name]['d']
        a, b, c = normal
        if abs(c) > 1e-6:
            zz = (-a * xx - b * yy - d) / c
            ax.plot_surface(xx, yy, zz, alpha=alpha, color=color, label=label)

    ax.set_xlabel('X coordinate (Å)')
    ax.set_ylabel('Y coordinate (Å)')
    ax.set_zlabel('Z coordinate (Å)')
    ax.set_title('Atoms and Calculated Best-Fit Planes')

    fake_patches = [plt.Rectangle((0, 0), 1, 1, fc=color, alpha=alpha + 0.3) for color, alpha, _ in
                    planes_to_plot.values()]
    legend_labels = [label for _, _, label in planes_to_plot.values()]
    ax.legend(handles=fake_patches, labels=legend_labels)

    plt.show()


# --- USAGE EXAMPLE ---
if __name__ == "__main__":
    all_atomic_coords = np.array([
        [1.0, 1.1, 5.05], [3.0, 0.9, 5.01], [0.9, 3.1, 4.95],
        [3.1, 3.0, 4.99], [5.5, 2.0, 8.30], [2.0, 2.0, 0.50]
    ])
    selected_atom_indices = [0, 1, 2, 3]
    selected_points = all_atomic_coords[selected_atom_indices]
    separation_distance = 1.5
    plane_results = find_parallel_planes(selected_points, separation_distance)
    plot_atoms_and_planes(all_atomic_coords, selected_points, plane_results, separation_distance)