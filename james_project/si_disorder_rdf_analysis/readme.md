# Lattice Disorder & RDF Visualizer

This script systematically introduces varying levels of atomic disorder into a crystalline structure and visualizes the structural degradation using the Radial Distribution Function (RDF). By tracking the broadening and loss of coordination shell peaks, it provides a quantitative way to identify the threshold where an ordered crystal loses its long-range order and transitions into an amorphous state.

## Features
* **Format Agnostic (via ASE):** Primarily configured to read standard LAMMPS `.data` files, but easily adaptable to any format supported by the Atomic Simulation Environment (ASE).
* **Systematic Perturbation:** Applies random Gaussian displacements to atomic positions across a user-defined percentage range (relative to the equilibrium bond length).
* **Automated RDF Analysis:** Computes and vertically offsets the Radial Distribution Function for each generated structure in a single visualization.
* **Phase Boundary Identification:** Visually tracks the washout of distinct secondary and tertiary coordination shells to pinpoint phase breakdown.

## Dependencies
This tool relies on the Atomic Simulation Environment (ASE) for structural manipulation and analysis, alongside standard scientific plotting libraries.

* `ase`
* `numpy`
* `matplotlib`

Install dependencies via pip:
```bash
pip install ase numpy matplotlib