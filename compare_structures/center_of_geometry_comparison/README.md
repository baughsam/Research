# VASP Intermolecular COG Distance Projector

A Python tool for calculating the Center of Geometry (COG) distance between two molecules within a Cartesian VASP/POSCAR file. This script extracts the connecting vector between the two molecules and projects it onto both the Cartesian axes ($X, Y, Z$) and the crystallographic lattice vectors ($\hat{a}, \hat{b}, \hat{c}$).

## Features
*   **Agnostic to Local Vibrations:** Uses the unweighted Center of Geometry (COG) to measure bulk molecular translation, preventing individual atomic wiggles (like terminal hydrogens) from skewing the intermolecular distance metric.
*   **Dual Projections:** Outputs absolute distances, Cartesian projections ($X, Y, Z$), and crystallographic projections ($a, b, c$) simultaneously.
*   **Iterative CSV Logging:** Designed to run iteratively. Appends results to a formatted CSV, making it easy to build a continuous dataset across multiple displacement files.

## Requirements
*   Python 3.x
*   NumPy

## Usage

### 1. Prepare Your Files
Ensure your input file is a VASP coordinate file (e.g., `POSCAR.vasp`) with explicitly defined Cartesian coordinates in Angstroms. The script reads the lattice vectors from lines 3-5 and the atomic coordinates starting at line 9.

### 2. Configure the Script
Open the Python file and navigate to the `__main__` execution block at the bottom. 
Define your file paths and the 0-based index lists for the two molecules you wish to compare. 

```python
if __name__ == "__main__":
    filename = 'your_displaced_structure_cartesian.vasp'
    output_filename = 'intermolecular_cog_distances.csv'
    
    # 0-based indices for the atoms in each molecule
    # Example: 22 carbon atoms for each molecule (pentacene)
    mol_1_carbons = list(range(0, 22))
    mol_2_carbons = list(range(22, 44))
    
    # Run calculation with a custom label (e.g., the specific phonon mode)
    calculate_and_export_cog_distances(
        filepath=filename, 
        mol1_indices=mol_1_carbons, 
        mol2_indices=mol_2_carbons, 
        output_csv=output_filename,
        label="Phonon_Mode_1_Displaced"
    )