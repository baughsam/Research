import os
import random
import numpy as np
from ase.io import read, write

# Inputs
input_file = 'si_54.data'
total_structures = 50 # Can now be any number (e.g., 8, 50, 500)
perc_min = 0  # Minimum disorder percentage
perc_max = 8  # Maximum disorder percentage
output_dir = 'si_training_set'
base_atom_format = 'lammps-data'


# Script
def generate_training_set():
    os.makedirs(output_dir, exist_ok=True)

    # Load the base crystalline structure
    base_atoms = read(input_file, format=base_atom_format, style='atomic')

    # Automatically detect the nearest-neighbor bond length
    dists = base_atoms.get_all_distances(mic=True)
    np.fill_diagonal(dists, np.inf)  # Ignore self-distances
    auto_bond_length = np.min(dists)

    print(f"Auto-detected baseline bond length: {auto_bond_length:.3f} Å")

    # Create dynamic bins evenly distributed across the total number of structures
    # e.g., 50 structures = 50 intervals defined by 51 edges
    edges = np.linspace(perc_min, perc_max, total_structures + 1)

    print(f"Targeting {total_structures} total structures evenly distributed from {perc_min}% to {perc_max}%...")

    for i in range(total_structures):
        lower_bound = edges[i]
        upper_bound = edges[i + 1]

        # Randomize the exact percentage within this specific continuous interval
        p = random.uniform(lower_bound, upper_bound)

        # Clone base structure and calculate standard deviation in Angstroms
        atoms = base_atoms.copy()
        stdev = auto_bond_length * (p / 100.0)

        # Apply Gaussian rattle with a random seed to ensure uniqueness
        seed = random.randint(1, 10000000)
        atoms.rattle(stdev=stdev, seed=seed)

        # Wrap coordinates back into the periodic box to ensure LAMMPS compatibility
        atoms.wrap()

        # Save the structure with a padded index and its exact percentage in the filename
        filename = f"{output_dir}/struct_{i + 1:03d}_{p:.2f}pct.data"
        write(filename, atoms, format='lammps-data', atom_style='atomic', units='metal')

        print(f"  -> Generated struct_{i + 1:03d} in range {lower_bound:.2f}% - {upper_bound:.2f}% (Exact: {p:.2f}%)")

    print(f"\nSuccessfully generated {total_structures} structures in the '{output_dir}' directory.")


if __name__ == "__main__":
    generate_training_set()