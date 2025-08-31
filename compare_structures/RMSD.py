# main.py
import os
import numpy as np
from pymatgen.core import Structure
from pymatgen.io.vasp import Poscar  # Import the Poscar class
import csv

def calculate_rmsd(structure1, structure2):
    """
    Calculates the Root Mean Square Deviation (RMSD) between two structures.

    This function assumes the atoms in both structures are in the same order.
    It correctly handles periodic boundary conditions when calculating distances.

    Args:
        structure1 (pymatgen.core.Structure): The first structure object.
        structure2 (pymatgen.core.Structure): The second structure object.

    Returns:
        tuple: A tuple containing:
            - float: The overall RMSD value in Angstroms.
            - list: A list of individual atomic displacements in Angstroms.

    Raises:
        ValueError: If the structures have a different number of atoms.
    """
    # --- Pre-computation Checks ---
    # Check if the number of atoms is the same in both structures
    if len(structure1) != len(structure2):
        raise ValueError("Error: The two structures have different numbers of atoms.")

    # A friendly warning if the chemical species do not match perfectly.
    # For a simple RMSD, we assume the atom order is conserved.
    if structure1.species != structure2.species:
        print("Warning: The species in the two structures are not identical or are in a different order.")

    # --- Calculation ---
    displacements = []
    squared_displacements = []

    # Iterate through each atom index in the structure
    for i in range(len(structure1)):
        site1 = structure1[i]
        site2 = structure2[i]

        # The site.distance() method in pymatgen automatically finds the
        # minimum distance across periodic boundaries using the lattice
        # information stored in the structure object.
        distance = site1.distance(site2)

        displacements.append(distance)
        squared_displacements.append(distance ** 2)

    # Calculate the mean of the squared displacements
    mean_squared_displacement = np.mean(squared_displacements)

    # The RMSD is the square root of the mean squared displacement
    rmsd = np.sqrt(mean_squared_displacement)

    return rmsd, displacements


def structure_from_content(content):
    """
    Helper function to create a pymatgen Structure from a VASP string.

    Args:
        content (str): A string containing the VASP structure (CONTCAR/POSCAR format).

    Returns:
        pymatgen.core.Structure: The parsed structure object.
    """
    # Use Poscar.from_str to parse the VASP structure content directly.
    # This is more robust as it explicitly defines the format and avoids
    # the need for temporary files.
    poscar = Poscar.from_str(content)
    return poscar.structure

def save_displacements_to_csv(displacements, structure, filename):
    """
    Saves the list of atomic displacements to a CSV file.

    Args:
        displacements (list): A list of displacement floats.
        structure (pymatgen.core.Structure): The reference structure to get element data.
        filename (str): The name of the output CSV file.
    """
    with open(filename, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        # Write the header
        csv_writer.writerow(['Atom Index', 'Element', 'Displacement'])
        # Write the data rows
        for i, displacement in enumerate(displacements):
            element = structure[i].species_string
            csv_writer.writerow([i + 1, element, f"{displacement:.6f}"])
    print(f"\nDisplacement data successfully saved to '{filename}'")

def read_contcar_file(filepath):
    with open(filepath, 'r') as f:
        return f.read()

QE_struc = structure_from_content(read_contcar_file("CONTCAR_QE_pent"))
VASP_struc = structure_from_content(read_contcar_file("CONTCAR_VASP_pent"))
np.set_printoptions(legacy='1.25')
B = calculate_rmsd(QE_struc, VASP_struc)

save_displacements_to_csv(B[1], QE_struc, "displacments_QE_VASP.csv")

print(B)
