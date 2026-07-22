# BerkeleyGW Exciton Energy & Velocity Pipeline

This directory contains the preprocessing pipeline required to map *ab initio* exciton energy dispersions from BerkeleyGW (computed on the SCC) into 3D macroscopic group velocities for semi-classical transport simulations.

## Overview
Because BerkeleyGW calculates finite-momentum ($Q \neq 0$) excitons in isolated `QXXXX` directories, the raw `eigenvalues_b1.dat` files must be parsed, stitched back into a 3D Brillouin zone grid, and differentiated. 

To avoid heavily burdening the cluster's file system or fighting with terminal-based debugging, this workflow is decoupled into two steps:
1. **Cluster Extraction:** Scrapes the data and strictly enforces the momentum ordering from `xctph.h5`.
2. **Local Processing:** Rebuilds the 3D grid and calculates the spatial gradients (velocities) locally.

---

## Step 1: Cluster Extraction (SCC)
Run `scc_extract_energies.py` directly on the cluster inside your main finite-$Q$ directory (e.g., `02-xct-Q/`). 

### Required Files in Directory:
* `xctph.h5` - The master scattering tensor.
* `QXXXX/03-singlet/eigenvalues_b1.dat` - The completed absorption outputs for all $Q$-points.

### Architecture Note: The "Glob Guarantee"
You do not need to provide a `bands.in` file to cross-reference coordinates. When BerkeleyGW generates the `xctph.h5` file, the `write_xct_h5.py` utility strictly reads the directories using `np.sort(glob.glob(...))`. Therefore, we are mathematically guaranteed that the $n$-th momentum state in the `.h5` ledger corresponds exactly to the folder named `Q(n)`. This alphabetical 1:1 mapping bypasses any bugs associated with coordinate precision or periodic boundary conditions (e.g., $-0.25$ vs $0.75$).

### Execution:
```bash
python scc_extract_energies.py
```
*Note: You can change the `target_state` variable at the top of the script to extract higher-order excitons.*

### Failsafes:
The script is designed for high-throughput task arrays. It will immediately **abort (sys.exit(1))** if it detects:
1. A missing `eigenvalues_b1.dat` file (e.g., a missing folder).
2. An incomplete file (e.g., a job wall-timed out before writing the target state).

### Output:
Generates `ordered_raw_energies_state_X.npz`. **Download this file to your local machine.**

---

## Step 2: Local Processing (PyCharm/Local)
Place `ordered_raw_energies_state_X.npz` in your local project directory and run `local_calc_velocities.py`.

### Execution:
```bash
python local_calc_velocities.py
```

### What it does:
1. Determines the exact $(N_x, N_y, N_z)$ dimensions of your reciprocal space grid from the master coordinates.
2. Maps the 1D ordered energies into a 3D NumPy tensor.
3. Computes the 3D spatial gradient using a 2nd-Order Central Difference scheme (`np.gradient`).
4. Converts the gradients into velocities ($v_g = \frac{1}{\hbar} \nabla_Q E$).
5. Flattens the $v_x, v_y, v_z$ matrices back into 1D arrays using the exact sequence dictated by the master ledger.

### Output:
Generates `final_velocity_payload.npz`. 
This file can be loaded directly into the `CentralDifference3d` advection solver, guaranteeing that the velocities natively match the shape and order of the transition matrices in the `xctph.h5` file.