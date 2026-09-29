# Disordered Structure Training Set Generator

This script automates the generation of a structural dataset featuring varying levels of atomic disorder. It systematically perturbs an ideal crystalline lattice across a defined continuous percentage range, creating evenly distributed structures. This is ideal for generating training data for machine learning models (such as Random Forest classifiers) trained to identify phase transitions and structural washout using Radial Distribution Function (RDF) descriptors.

## Key Features

* **Dynamic Even Distribution:** Divides the total target percentage range (`perc_min` to `perc_max`) into evenly spaced, continuous intervals based on the requested number of structures. Guarantees a flat distribution without binning remainders or edge-case gaps.
* **Auto-Detected Bond Length:** Automatically scans the input structure to calculate the minimum nearest-neighbor distance (ignoring periodic self-interactions), establishing an accurate baseline for percentage-based standard deviations.
* **Continuous Randomization:** Rather than locking to integer percentages, the script draws a random float uniformly from within each interval to prevent artificial clustering.
* **LAMMPS I/O Safety:** Wraps perturbed coordinates back into the periodic cell boundaries before export, preventing wrapped-coordinate inflation artifacts when downstream tools read the non-orthogonal box.

## Dependencies

This script relies on the Atomic Simulation Environment (ASE) and NumPy.

pip install ase numpy

## Configuration

Open the script to adjust the core variables under the Input header:

* `input_file`: Path to the perfectly relaxed base cell.
* `total_structures`: The exact number of structures to generate.
* `perc_min` / `perc_max`: The lower and upper bounds for structural perturbation. 
  * *Note: A maximum around `8.0` is generally sufficient to cross the crystalline-to-amorphous threshold for materials like diamond-cubic silicon.*
* `output_dir`: The directory where the dataset will be saved.
* `base_atom_format`: a flag for ase in case we aren't using lammps

## Usage

Execute the script via Python:

python generate_training_set.py

## Output

The script generates a directory (default: `training_set/`) populated with LAMMPS data files. Each file is sequentially numbered and stamped with its exact perturbation percentage for easy parsing by downstream featurization scripts:

training_set/

├── struct_001_0.14pct.data

├── struct_002_0.31pct.data

├── struct_003_0.45pct.data

...

└── struct_050_7.92pct.data