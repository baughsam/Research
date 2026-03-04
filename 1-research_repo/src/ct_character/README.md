# CT Characterization (ct_character)

A Python package for analyzing Charge Transfer (CT) excitons in molecular crystals. This tool processes volumetric data (Gaussian `.cube` files), applies geometric masks, and calculates rigorous, volume-corrected physical properties like the Charge Transfer Ratio.

## Features

* **Volumetric Analysis:** Reads standard Gaussian `.cube` files representing excited-state probability densities ($|\Psi|^2$).
* **Geometric Masking:** Isolates probability mass within specified shapes:
    * `EllipticalCylinder`: Defined by standard radial axes and length.
    * `Parallelepiped`: A skewed 3D box strictly defined by the crystal lattice vectors.
* **Physical Observables:**
    * **Charge Transfer (CT) Ratio:** The rigorous, volume-corrected fraction of the exciton localized within the geometric boundary.
    * **1D Radial Analysis:** Evaluates the probability density as a function of distance from the center.
* **Performance:** Utilizes a custom **Memory-Optimized Z-Plane Slicing** algorithm, allowing the analysis of massive density grids without overloading system RAM.

## Installation
```bash
pip install "git+[https://github.com/baughsam/Research.git#subdirectory=1-research_repo](https://github.com/baughsam/Research.git#subdirectory=1-research_repo)"
```
*(Note: Be sure to update the URL above to point to the new lab repository once deployed!)*

## Usage
Once installed, you can execute the physics engine from any directory using Python's module flag. This is the standard, most robust way to run the analysis:

```bash
python -m ct_character.main path/to/input.in [--print-analysis-graph] [--print-mask-cube]
```

**Shortcut Command:**
Once you are comfortable with the module structure above, the installation also provides a global terminal shortcut for convenience. You can run the exact same analysis by simply typing:
```bash
ct-analysis path/to/input.in [--print-analysis-graph] [--print-mask-cube]
```

**Optional Flags:**
* `--print-analysis-graph`: Automatically generate and save a `.png` plot of the normalized, volume-corrected probability density.
* `--print-mask-cube`: Generate a `_MASK.cube` file that visualizes the 3D geometric shape used for integration (useful for debugging your shape orientation in VESTA or Avogadro).

### Input File Format
The input file is a strictly formatted 7-line text file (e.g., `INPUT_CTCALC.in`).

**Example:**
```text
PEN1_3D-xyz.cube
EllipticalCylinder
5.1420
5.1420
28.1257
PEN1_Results
True
```

**Line-by-Line Guide:**
1. **Cube Filename:** Path to the `.cube` file (relative to the input file).
2. **Shape Model:** Must be `EllipticalCylinder` or `Parallelepiped` (or `Box`).
3. **Dimension/Vector A:** X-axis radius (Cylinder) OR Vector A coordinates separated by spaces (Parallelepiped).
4. **Dimension/Vector B:** Y-axis radius (Cylinder) OR Vector B coordinates separated by spaces (Parallelepiped).
5. **Dimension/Vector C:** Z-axis length (Cylinder) OR Vector C coordinates separated by spaces (Parallelepiped).
6. **Output Prefix:** String used to name the generated output files.
7. **1D Analysis Toggle:** `True` or `False`. Toggles the calculation and output of the shell-by-shell distance data.

## Outputs
Depending on your input toggles and flags, the code generates up to five files in the same directory as the input:

1. **`{Prefix}_OUT.txt`**: (Always generated) A human-readable summary report containing grid dimensions, voxel volumes, the Wavefunction Norm, and the final Charge Transfer Ratio.
2. **`{Cube_Filename}_density.npy`**: (Always generated) A binary numpy cache of the parsed volumetric data. This is generated on the first run and significantly speeds up parsing time on subsequent runs analyzing the same `.cube` file.
3. **`{Prefix}_1D-distance-involume.dat`**: (If Line 7 is `True`) A data file containing the raw, legacy, and probability density metrics binned by radial shell distance.
4. **`{Prefix}_Volume_Corrected_Density.png`**: (If `--print-analysis-graph` is passed) A graphical plot of the scaled, volume-corrected volumetric density.
5. **`{Prefix}_MASK.cube`**: (If `--print-mask-cube` is passed) A 3D `.cube` file rendering the boolean geometry mask for visualization.

## Physics Implementation Details

Because the volumetric `.cube` data represents the electron-hole correlation function, $|\Psi|^2$, we evaluate the spatial distribution strictly in terms of probability mass. The engine applies a rigorous volume-correction to avoid geometrical integration artifacts:

* **Volume-Correction:** Standard cumulative integration naturally accumulates larger values at greater distances simply because expanding radial shells geometrically enclose more volume (scaling with $r^2$). To remove this geometric expansion effect, the probability mass within each discrete radial shell, $\Delta C(r_i)$, is divided by the exact number of grid voxels in the specific shell $N(r_i)$, before the cumulative sum is performed.
* **Normalization:** By dividing by the voxel count, we convert the summation from a conserved total mass into a sum of local per-voxel averages. The code then applies a final scaling normalization to force the arrays to plateau at their true theoretical maximums ($1.0$ for probability fraction, or $1.0/dV$ for volumetric density).
* **CT Ratio:** The final Charge Transfer Ratio reported by the code is calculated directly from the plateau of this volume-corrected, normalized probability metric, ensuring the value represents the true physical concentration of the exciton within the specified domain.

---
*Based on the original Fortran implementation by Sahar Sharifzadeh & Pierre Darancet (The Molecular Foundry, Berkeley). Python Rewrite & Physics Optimization by Samson Baughman.*