# GW Band Structure Plotter

**Script Name:** `gw_bandstructures.py`  
**Description:** A Python utility for plotting and comparing electronic band structures from computational materials science codes (specifically tailored for BerkeleyGW output formats). It visualizes Quasiparticle (GW) energies alongside Mean-Field (LDA/DFT) energies, automatically aligns the Valence Band Maximum (VBM) to zero, and calculates band gaps.

---

### 1. Features
* **Multi-Dataset Comparison:** Overlays multiple band structure files (e.g., Pristine vs. Phonon-perturbed/Temperature-dependent) on a single plot.
* **Automatic Alignment:** Shifts energy scales so the VBM (defined by index `nv`) is at 0 eV.
* **Gap Calculation:** Automatically computes and displays the $E_g$ (LUMO - HOMO) in the legend for both LDA and GW.
* **Customizability:** Toggles for plotting LDA lines, custom colors, energy windows, and High-Symmetry K-point labels.

---

### 2. Dependencies
To run this script, you need a standard Python 3 scientific stack:
* `numpy`
* `matplotlib`
