# Context Transfer: Multi-Band Exciton BTE Framework

## 1. High-Level Objective

The goal of this project is to model the ultrafast spatiotemporal propagation and momentum-space relaxation of excitons in crystalline organic semiconductors using a semiclassical Boltzmann Transport Equation (BTE) parameterized by first-principles (GW-BSE + DFPT) calculations.

Recently, the codebase transitioned from a hardcoded 1-band model (where secondary states acted as infinite sinks) to a dynamically scalable **N-band modular architecture**. This allows the simultaneous simulation of multiple interacting electronic states and properly enforces detailed balance via back-scattering.

## 2. Photophysical Context: Naphthalene

The target system is crystalline naphthalene.

* **State Ordering:** Naphthalene exhibits an inverted state ordering relative to longer acenes like pentacene. The lowest-lying singlet state ($S_1$) is the optically forbidden (dark) $1L_b$ state. The higher-energy $S_2$ state is the strongly absorbing (bright) $1L_a$ state.

* **Exciton-Phonon Scattering Asymmetry:** Exciton relaxation between these states is thermodynamically asymmetric:

  * **Downward Relaxation (**$S_2 \to S_1$**):** Exothermic process. Scales with phonon emission ($1 + n_{q\nu}$), accounting for spontaneous emission and driving rapid population decay into the dark state.

  * **Upward Back-Scattering (**$S_1 \to S_2$**):** Endothermic process. Relies on thermal energy and scales purely with phonon absorption ($n_{q\nu}$).

* This asymmetry is critical. The BTE must explicitly model both forward and backward rates to capture the true thermalization of the exciton population across the $S_1$ and $S_2$ bands.

## 3. Mathematical Foundation (Coupled BTE)

The core simulation solves a system of coupled transport equations for $N$ bands. The time-evolution of the exciton density $n$ in state $a$ at momentum $Q$ and position $R$ is:

$$
\frac{\partial n_a(Q,R)}{\partial t} = -\nabla_R \cdot (v_a(Q) n_a(Q,R)) + K_{scat}[n(Q,R)]
$$

The scattering operator $K_{scat}$ for a given band $a$ computes all transitions into and out of the state across all $N$ bands (where $b$ is the interacting band, and $q$ is the phonon momentum):

$$
K_{scat}[n_a] = \sum_{b, q} k_{b(Q+q) \to a(Q)} n_b(Q+q, R) - \sum_{b, q} k_{a(Q) \to b(Q+q)} n_a(Q, R) - k_a^{rad} n_a(Q, R)\delta_{Q,\Gamma}
$$

## 4. Codebase Architecture (N-Band Modular Framework)

To compute the BTE across multiple bands simultaneously without hardcoding logic, the data structure was upgraded from a 3D tensor to a **4D phase-space tensor**: `(N_bands, Nx, Ny, Nq)`.

The framework relies on decoupling the *physical state* from the *mathematical integrators*. The solvers simply loop over the `N_bands` axis, completely agnostic to whether they are solving a 1-band, 2-band, or 10-band system.

## 5. Component Breakdown

### A. Configuration (`config.txt` & `io.py`)

The system is configured via dynamic tuples in the input file, allowing arbitrary scaling:

```
state_names = ('S2', 'S1')          # Defines band indices 0 and 1
state_advection = (True, False)     # Toggles spatial movement per state
state_gamma_decay = (0.005, 0.0)    # Radiative decay rates per state

```

### B. Core Driver (`main.py`)

* Dynamically parses the configuration tuples.

* Initializes the 4D `n_initial_4d` tensor, injecting the initial wavepacket strictly into the 0th index band ($S_2$).

* Dynamically builds a list of independent spatial solvers (`spatial_solvers = [Solver_S2, Solver_S1]`). If advection is disabled for a state, its velocity arrays are zeroed out.

* Queries the `.npz` payload using the exact state names (e.g., searches for `Rate_S2_S1`) to construct the inter-band mapping dict.

### C. Scattering Operator (`k_scat.py`)

* Contains the `MultiBandTransitionMatrix` class.

* Constructs an $N \times N$ block tensor (`K_tensor`) containing the fully momentum-resolved transition rates for every permutation of states.

* Calculates the entire $K_{scat}$ term for the 4D tensor in a single, highly optimized, dimension-agnostic NumPy operation:
  `np.einsum('abij, bxyj -> axyi', self.K_tensor, array_4d)` (where `a`=band_out, `b`=band_in, `i`=q_out, `j`=q_in).

### D. Time Integrator (`NumMthds.py`)

* Contains `MultiBandRungeKutta4`.

* Calculates the global safe time step (CFL condition) checking max velocities and max scattering rates across *all* bands.

* Uses a helper function `_calc_advection_4d` to iterate over the `N_bands` axis, applying each band's distinct spatial solver to its respective 3D slice.

* Features a `solve_hdf5_compressed` method to stream the massive 4D history tensor directly to disk, preventing RAM overflow.

### E. Post-Processing (`analysis.py`)

* All visualization functions (GIF exporters, diffusion extractors, momentum projectors) begin by collapsing the 4D tensor to 3D to visualize total macroscopic density: `np.sum(frames[i], axis=0)`.

### F. Data Preparation (`compile_scat_rates.py`)

* Standalone script that extracts the raw 5D $g$-tensor from BerkeleyGW (`xctph.h5`) and integrates out the phonon modes ($\nu$).

* Applies Fermi's Golden Rule with exact phase-space kinematics, weighting the transitions using the Bose-Einstein distribution (temperature-dependent) and a Gaussian energy conservation function.

* Explicitly computes all necessary permutations for Naphthalene ($S_2 \to S_2$, $S_2 \to S_1$, $S_1 \to S_2$, $S_1 \to S_1$) and packages them into the `.npz` payload expected by `main.py`.

* Includes utility functions to extract 1D cuts of the scattering paths for explicit comparison of forward vs. backward scattering strengths.