import numpy as np
import warnings
import os
import csv


def parse_vasp_and_lattice(filepath):
    """
    Parses a Cartesian POSCAR (VASP) file.
    Returns the coordinates, total atom count, and lattice vectors.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Could not find {filepath}")

    with open(filepath, 'r') as f:
        lines = f.readlines()

    # Line index 1 is the universal scaling factor
    scale = float(lines[1].strip())

    # Lines 2, 3, 4 are the lattice vectors
    a_vec = np.array([float(x) for x in lines[2].split()]) * scale
    b_vec = np.array([float(x) for x in lines[3].split()]) * scale
    c_vec = np.array([float(x) for x in lines[4].split()]) * scale

    # Line 6 contains the number of atoms per species
    counts = [int(x) for x in lines[6].split()]
    total_atoms = sum(counts)

    # Coordinates start at line index 8
    coords = []
    for line in lines[8:8 + total_atoms]:
        parts = line.split()
        if len(parts) >= 3:
            coords.append([float(parts[0]), float(parts[1]), float(parts[2])])

    return np.array(coords) * scale, total_atoms, a_vec, b_vec, c_vec


def calculate_and_export_cog_distances(filepath, mol1_indices, mol2_indices, output_csv, label="Equilibrium"):
    """
    Calculates COG distances in Cartesian (x,y,z) and lattice (a,b,c) projections.
    Exports the results to a structured CSV file.
    """
    coords, total_atoms, a_vec, b_vec, c_vec = parse_vasp_and_lattice(filepath)

    # Validation check
    for idx in mol1_indices + mol2_indices:
        if idx < 0 or idx >= total_atoms:
            warnings.warn(f"Warning: Index {idx} is out of bounds (Total atoms: {total_atoms}).")
            return

    # Calculate COGs
    cog1 = np.mean(coords[mol1_indices], axis=0)
    cog2 = np.mean(coords[mol2_indices], axis=0)

    # Connecting vector
    r12 = cog2 - cog1
    total_dist = np.linalg.norm(r12)

    # Cartesian X, Y, Z distances (absolute differences)
    dist_x = abs(r12[0])
    dist_y = abs(r12[1])
    dist_z = abs(r12[2])

    # Normalize lattice vectors to get direction (unit vectors)
    a_hat = a_vec / np.linalg.norm(a_vec)
    b_hat = b_vec / np.linalg.norm(b_vec)
    c_hat = c_vec / np.linalg.norm(c_vec)

    # Project connecting vector onto lattice vectors
    dist_a = abs(np.dot(r12, a_hat))
    dist_b = abs(np.dot(r12, b_hat))
    dist_c = abs(np.dot(r12, c_hat))

    # Prepare data row
    row_data = [
        label,
        round(total_dist, 5),
        round(dist_x, 5), round(dist_y, 5), round(dist_z, 5),
        round(dist_a, 5), round(dist_b, 5), round(dist_c, 5)
    ]

    # Check if file exists to write headers
    file_exists = os.path.isfile(output_csv)

    with open(output_csv, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            # Write clear headers if it's a new file
            writer.writerow([
                "System_State", "Total_Dist(Å)",
                "dX_Cartesian(Å)", "dY_Cartesian(Å)", "dZ_Cartesian(Å)",
                "dA_Projected(Å)", "dB_Projected(Å)", "dC_Projected(Å)"
            ])
        writer.writerow(row_data)

    print(f"Successfully appended {label} distances to {output_csv}")


if __name__ == "__main__":
    # Define inputs
    filename = 'pristine_pent.vasp'
    output_filename = 'intermolecular_cog_distances.csv'

    # Define your molecules (e.g., the 22 carbons for Mol 1 and Mol 2)
    mol_1_carbons = [35,33,31,37,39,29,27,41,44,23,24,26,43,42,28,30,40,38,32,34,36,25]
    mol_2_carbons = [12,14,16,10,18,8,20,6,21,4,1,2,3,22,5,19,7,17,9,15,11,13]

    # Run the calculation
    # You can change the label to "Phonon_Mode_1", "Phonon_Mode_2", etc. as you iterate
    calculate_and_export_cog_distances(
        filepath=filename,
        mol1_indices=mol_1_carbons,
        mol2_indices=mol_2_carbons,
        output_csv=output_filename,
        label="Pristine Pent"
    )