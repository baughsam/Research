# CT Characterization (ct_character)

A Python package for analyzing Charge Transfer (CT) excitons in molecular crystals. This tool processes volumetric data (Gaussian `.cube` files), applies geometric masks (Cylindrical or Ellipsoidal), and calculates physical properties like Charge Transfer Ratios, Dipole Moments, and Radial Distribution Functions.

## Features

* **Volumetric Analysis:** Reads standard Gaussian `.cube` files.
* **Geometric Masking:** Isolates electron density within a specified `Cylinder` or `Ellipsoid`.
* **Physical Observables:**
    * **CT Ratio:** Fraction of charge contained within the geometric volume.
    * **Multipoles:** Calculates Dipole ($\vec{\mu}$) and Quadrupole ($Q$) moments.
    * **Radial Distribution Function (RDF):** Analysis of charge density vs. distance from center.
    * **Average Radius:** Expectation value $\langle r \rangle$ of the exciton size.
    * **Projections:** 1D Planar averages along lattice vectors ($\vec{a}, \vec{b}, \vec{c}$).
* **Performance:** Uses vectorized NumPy operations for speed (replacing legacy Fortran loops).

## Installation
pip install "git+https://github.com/baughsam/Research.git#subdirectory=1-research_repo"

## Usage
Once installed, the package provides a command-line tool ct-analysis. You can run it from any directory.
ct-analysis path/to/input.in

**Input File Format**
The input file is a strictly formatted 6-line text file (e.g., INPUT_CTCALC.in).

Example:
PEN1_3D-xyz.cube
Cylinder
5.1420
5.1420
28.1257
PEN1_Results

Line-by-Line Guide:
1. Cube Filename: Path to the .cube file (relative to the input file).
2. Shape Model: Must be Cylinder or Ellipsoid.
3. Dimension A: Radius (or semi-axis) along X-axis (in Bohr).
4. Dimension B: Radius (or semi-axis) along Y-axis (in Bohr).
5. Dimension C: Length (or semi-axis) along Z-axis (in Bohr).
6. Output Prefix: String used to name the output files.

## Outputs
The code generates two output files in the same directory as the input:
1. {Prefix}_OUT.txt: A human-readable summary report containing grid dimensions, total volume, calculated dipoles, and average radii.
2. {Prefix}_stats.json: A machine-readable JSON file containing raw statistics (useful for plotting or batch analysis).

## Project Structure
src/ct_character/
├── main.py         # Entry point (CLI logic)
├── Solver.py       # Physics engine (Normalization, Multipoles, RDF)
├── IOHandler.py    # File I/O (.cube parsing, Report writing)
├── Exciton.py      # Data classes (Configuration, ExcitonData)
└── Shape.py        # Geometric logic (is_inside checks)

## Physics Implementation Details
- Normalization: The density grid is normalized such that $\sum \rho = 1.0$ (probability distribution) before calculating multipoles and projections.
- Coordinates: Grid indices $(i,j,k)$ are transformed to Cartesian coordinates $(x,y,z)$ using the lattice vectors read from the cube header.
- CT Ratio: Defined as $\frac{\sum \rho_{\text{inside}}}{\sum \rho_{\text{total}}}$.
  - Paper: [Low-Energy Charge-Transfer Excitons in Organic Solids from First-Principles: The Case of Pentacene](https://pubs-acs-org.ezproxy.bu.edu/doi/10.1021/jz401069f)
