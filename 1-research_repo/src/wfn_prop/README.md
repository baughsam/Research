# Wavefunction Propagation Framework (`wfn_prop`)

The `wfn_prop` package is a computational tool designed to simulate the ultrafast spatiotemporal propagation and momentum-space relaxation of photoexcited excitons in materials. The core engine models semiclassical exciton dynamics by solving transport and advection equations across discrete real-space coordinates and a first-principles momentum grid.

This framework provides a direct bridge between ab initio electronic structure calculations and macroscopically observable transport phenomena, based on the theoretical models established in:
> *Phonon-Driven Femtosecond Dynamics of Excitons in Crystalline Pentacene from First Principles*
> Cohen et al., **Physical Review Letters 132, 126902 (2024)**.

---

## What the Code Does

The package performs explicit time integration to track how an initial exciton population distribution spreads out in real space and thermalizes across different momentum states over time. It handles two primary physical mechanisms simultaneously:

1. **Advection (Spatial Drift):** Simulates the spatial movement of excitons driven by their physical group velocities (derived from the exciton bandstructure) using a 2nd-order Central Difference.
2. **Scattering (Phase Relaxation):** Models quantum transitions—both scattering within the same exciton band (intraband) and decaying to secondary states (interband)—mediated by an ab initio exciton-phonon transition rate matrix.

---

## Repository Structure

* **`main.py`**: The central command-line entrypoint that coordinates parsing config files, allocating grids, loading data payloads, and driving the simulation loop.
* **`NumMthds.py`**: Houses the numerical finite-difference solvers (`CentralDifference3d`, `UpwindDifference3d`) and the explicit 4th-order Runge-Kutta (`RungeKutta4`) time integrator.
* **`k_scat.py`**: Defines the physical scattering operators, including the primary `two_state_transition_matrix` class which constructs transition loss and gain matrices from raw physics payload inputs.
* **`io.py`**: Handles file system interactions, parsing key-value text configurations in an order-independent manner while filtering out comments.
* **`analysis.py`**: Contains the post-processing pipeline used to extract macroscopic diffusion constants and bake two-panel real/momentum space tracking GIFs.

---

## Simulation Inputs

To execute a calculation, the framework requires three structural items:

### 1. The Configuration File (`config.txt`)
A plaintext input file containing execution control parameters. Parameters can be specified in any arbitrary order:

| Parameter | Type | Description |
| :--- | :--- | :--- |
| `physics_file` | String | Path to the `.npz` file containing scattering matrices and mapping arrays. |
| `vel_file` | String | Path to the `.npz` file holding momentum-resolved group velocities. |
| `length_x`, `length_y` | Float/Int | Total physical dimensions of the simulation box in nanometers. |
| `grid_x`, `grid_y` | Integer | Spatial discretization grid dimensions (number of nodes). |
| `sim_time` | Float | Total simulation runtime length in femtoseconds. |
| `init_mode` | String | Initial distribution type: `"gauss"` (broad wavepacket) or `"single-q"` (isolated state). |
| `amplitude` | Float | Scaling multiplier for the peak density of the initial distribution. |
| `sigma_R` | Float | Real-space spatial standard deviation (spread) in nm. |
| `sigma_Q` | Float | Momentum-space standard deviation spread in nm⁻¹ (for `"gauss"` mode). |
| `target_q_index` | Integer | Explicit state index to hold 100% of the initial population (for `"single-q"` mode). |
| `gif_filename` | String | Baseline name for tracking data outputs. |

### 2. Ab Initio Data Payloads
* **Physics Payload (`.npz`):** Contains the raw momentum grid coordinates (`Qpts`), scattering matrices between states (`Rate_BB`, `Rate_BD`), radiative decay parameters, and state index maps.
* **Velocity Payload (`.npz`):** Contains matching momentum grid coordinates and the resolved electronic group velocity arrays along the x and y crystal axes (`vel_x`, `vel_y`).

---

## Simulation Outputs

Every completed run outputs two files to your working environment:

1. **`output.log`**: A complete text duplicate of the program execution. It prints a signature startup banner, lists every configuration parameter detected during parsing, tracks data loading status, and prints progress updates before logging `"Job Complete"`.
2. **`[gif_filename]_frames.npz`**: A compressed binary data archive containing the raw mathematical results of the simulation. It packages:
    * `frames`: A 4D array tracking the physical density over space (X, Y) and momentum state (Q) at each recorded time interval.
    * `dt`: The calculated stable numerical timestep size used by the integrator loop.

---

## How to Run

### Installation
Register the package workspace with your environment using your terminal:
```bash
pip install -e 
```

### Execution
Type the shortcut command and pass it the path to your parameter configuration text file:
```bash
wfn-run config.txt
```

### Post-Processing (Visualizing the Output)
Because the core code writes raw numerical data frames to disk to maximize processing speeds, you use a secondary local script to process the output archive into visible results:

```python
import numpy as np
from wfn_prop.analysis import export_diffusion_gif_updated

# 1. Load data from the simulation run
data = np.load("file_name_simulation_run_frames.npz")
frames = data['frames']
dt = data['dt']

# 2. Load the original momentum state vectors
physics = np.load("compiled_scat_rates_data_S0_S1_8x8x8.npz")
q_vectors = physics['Qpts']

# 3. Generate a two-panel animated GIF of spatial density and momentum projection
export_diffusion_gif_updated(
    frames=frames, dt=dt, save_interval=2,
    length_x=100, length_y=100, grid_x=250, grid_y=250,
    right_panel_mode='qgrid', q_vectors=q_vectors, projection=('x', 'y'),
    filename="exciton_dynamics.gif"
)
```