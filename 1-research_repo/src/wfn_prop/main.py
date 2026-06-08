import numpy as np
import scipy.constants as const
import matplotlib.pyplot as plt
from wfn_prop.NumMthds import RungeKutta4, UpwindDifference2d, UpwindDifference3d, CentralDifference3d
from wfn_prop.k_scat import Decay, FickDiff, PhononScat, two_state_transition_matrix
from wfn_prop.analysis import extract_diffusion_constant, visualize_simulation, export_diffusion_gif

# Gaussian Distribution Function
def gaussian_dist_2d(x_pos, y_pos, spread_x, spread_y, amplitude, center_x, center_y):
    term1 = - ( (x_pos - center_x)**2 ) / (2 * spread_x**2)
    term2 = - ( (y_pos - center_y)**2 ) / (2 * spread_y**2)
    dist_funct = amplitude * np.exp( term1 + term2 )
    return dist_funct

def initialize_transition_matrix(energies_eV: np.ndarray, temp_K: float, coupling_constant: float) -> np.ndarray:
    """
    Generates a transition matrix based on Fermi's Golden Rule and Bose-Einstein statistics.

    :param energies_eV: 1D array of energies in eV of length Nq
    :param temp_K: Temperature in Kelvin
    :param coupling_constant: The base transition rate |M_0|^2 / hbar in units of (fs^-1)
    :return: Time-Independent Transition Matrix
    """

    Nq = len(energies_eV)
    k_B = const.k * 6.242e18 # eV/K
    k_B_T = k_B * temp_K

    # Energy Difference Matrix
    # Broadcasting creates size (Nq, Nq) matrix
    dE = energies_eV[None, :] - energies_eV[:, None]

    # Eliminate floating point noise for degenerate states
    dE[np.abs(dE) < 1e-10] = 0.0

    #Bose-Einstein Statistics
    with np.errstate(divide='ignore', invalid='ignore'):
        N_phonons = 1.0 /  (np.exp(np.abs(dE) / k_B_T) - 1.0)

    #Initialize Scattering Matrix (empty)
    W = np.zeros((Nq, Nq))

    # Normalization of the transition rate to ensure that our discretization
    # of a continuous momentum space doesn't change the physics
    base_rate = coupling_constant / Nq

    # Fermi's Golden Rule (Absorption and Emission)
    # Upward Transition (Absorption)
    W[dE > 0] = base_rate * N_phonons[dE > 0]
    # Downward Transition (Emission)
    W[dE < 0] = base_rate * (N_phonons[dE < 0] + 1) # +1 = Spontaneous Emission due to  Heisenburg Uncetainty Principle (delta_E*delta_t >= hbar/2)

    #Particle Conversation (Dealing w/ the Diagonal)
    # - Excitons cannot scatter into their own states
    np.fill_diagonal(W, 0.0)
    # - Diagonal W[i,i] must represent the total rate of excitons leaving state i
    drain_rate = -np.sum(W, axis=1) # Sums everything across a row # np.ndarray
    np.fill_diagonal(W, drain_rate)

    return W


def initialize_transition_matrix_RTA(energies_eV: np.ndarray, temp_K: float, coupling_constant: float) -> np.ndarray:
    """
    Validation matrix strictly enforcing the Relaxation Time Approximation (RTA).
    Forces a constant scattering rate and perfectly thermalized momentum randomization.
    """
    Nq = len(energies_eV)
    k_B = const.k * 6.242e18 # eV/K
    k_B_T = k_B * temp_K

    # 1. Calculate the exact, constant macroscopic scattering rate from the derivation
    E_c = 0.05 # eV
    tau_0 = 1.0 / coupling_constant
    tau_T = tau_0 * np.tanh(E_c / (2 * k_B_T))
    Gamma_T = ( 1.0 / tau_T ) # This is the exact constant drain rate for all states

    # 2. Calculate the Thermal Equilibrium Distribution P(E)
    boltzmann_factors = np.exp(-energies_eV / k_B_T)
    P_eq = boltzmann_factors / np.sum(boltzmann_factors) # Normalized probabilities

    # 3. Build the RTA Transition Matrix
    W = np.zeros((Nq, Nq))
    for i in range(Nq):
        for j in range(Nq):
            if i != j:
                # Scatter into state j proportional to its thermal weight.
                # The denominator (1 - P_eq[i]) ensures that the sum of the row
                # strictly equals Gamma_T even though excitons cannot scatter into their own state.
                W[i, j] = Gamma_T * (P_eq[j] / (1.0 - P_eq[i]))

    # 4. Set the diagonal (drain rate) to perfectly conserve particles
    drain_rate = -np.sum(W, axis=1)
    np.fill_diagonal(W, drain_rate)

    return W

def generate_velocity_arrays_tight_binding(Nx: int, Ny: int, max_velocity: float = 1.0):
    if Nx != Ny:
        raise ValueError("Nx != Ny")

    qx = np.linspace(-np.pi, np.pi, Nx)
    qy = np.linspace(-np.pi, np.pi, Ny)

    Qx, Qy = np.meshgrid(qx, qy)

    Vx = np.sin(Qx) * max_velocity
    Vy = np.sin(Qy) * max_velocity

    Vx_1D = Vx.flatten()
    Vy_1D = Vy.flatten()

    return Vx_1D, Vy_1D



def generate_energy_array_harmonic(Nx: int, Ny: int, max_energy_eV: float=0.1) -> np.ndarray:
    if Nx != Ny:
        raise ValueError("Q_x != Q_y")
    num_of_q_states = Nx * Ny

    qx = np.linspace(-1, 1, Nx)
    qy = np.linspace(-1, 1, Ny)

    # Build the 2D Phase Space
    # meshgrid takes our 1D axes and creates a full 2D coordinate system
    Qx, Qy = np.meshgrid(qx, qy) # Creates 2 3x3 matrices

    # Calculate Energy (Parabolic Dispersion: E = Qx^2 + Qy^2)
    # The center (0,0) will be 0.0 eV. The edges will be higher energy.
    R_squared = Qx ** 2 + Qy ** 2

    # Normalize the parabola so the highest energy corners exactly equal max_energy_eV
    max_R_squared = np.max(R_squared)
    E_2D = (R_squared / max_R_squared) * max_energy_eV

    # Collapse Phase Space to a 1D list of states
    # This turns the (Nx, Ny) array into a flat array of length (Nx * Ny)
    E_1D = E_2D.flatten()

    # Sanity Check: Size of our 1d array is the correct size of our matrix
    if np.size(E_1D) != num_of_q_states:
        raise ValueError("Size of E_1D does not match number of physical Q-states.")

    return E_1D

def generate_energy_array_tight_binding(Nx: int, Ny: int, half_bandwidth_eV: float=0.1) -> np.ndarray:
    """
    Generates a 2D tight-binding energy dispersion mapped to a 1D array.
    E(k) = Delta * (1 - cos(kx)) + Delta * (1 - cos(ky))
    """
    if Nx != Ny:
        raise ValueError("Nx != Ny")

    # Use the exact same Brillouin zone bounds as your velocity array!
    qx = np.linspace(-np.pi, np.pi, Nx)
    qy = np.linspace(-np.pi, np.pi, Ny)

    # Build the 2D Phase Space
    Qx, Qy = np.meshgrid(qx, qy)

    # Calculate Energy using the Tight-Binding dispersion
    # Center (0,0) is 0.0 eV. Corners (+/- pi, +/- pi) are maximum energy (4 * Delta).
    E_2D = half_bandwidth_eV * (1 - np.cos(Qx)) + half_bandwidth_eV * (1 - np.cos(Qy))

    # Collapse Phase Space to a 1D list of states
    E_1D = E_2D.flatten()

    return E_1D


# Initializing Gaussian Wavepack on N_x x N_y sized grid
# Real Space Dimension in nanometers
length_x = 100#40
length_y = 100#40

# Grid Dimensions
grid_x = 250#100
grid_y = 250#100

# Simulation Distances
delta_x = length_x / (grid_x-1)
delta_y = length_y / (grid_y-1)

#Box (real space) center
x_0 = length_x / 2 #+ (10)
y_0 = length_y / 2 #+ (10)

# Gaussian Spread (Change to actual simulation values)
sigma_x = 2
sigma_y = 2

# Amplitude
amplitude = 1

# Initialize 2D Grid
occupation_matrix_2d = np.zeros((grid_x, grid_y))

# Momentum Space (currently an arbitrary 3x3 dimensional array)
Nx = Ny = 10 # Nx and Ny should be the same (at this point in time)
Nq = Nx * Ny

# Velocity Arrays (would be calculated form the exciton dispersion)
v_x_array, v_y_array = generate_velocity_arrays_tight_binding(Nx=Nx, Ny=Ny, max_velocity=0.5) # v ~ nm/fs

# Initialize 3D array
occupation_matrix_3d = np.zeros((grid_x, grid_y, Nq))

for i in range(grid_x):
    for j in range(grid_y):
        x_i = i * delta_x
        y_i = j * delta_y
        occupation_matrix_3d[i,j, 13] = gaussian_dist_2d(x_pos=x_i, y_pos=y_i, spread_x=sigma_x, spread_y=sigma_y, amplitude=amplitude, center_x=x_0, center_y=y_0)



# 1. Recreate the X and Y grids for plotting (matching your loop logic)
X_grid = np.empty((grid_x, grid_y))
Y_grid = np.empty((grid_x, grid_y))
for i in range(grid_x):
    for j in range(grid_y):
        X_grid[i, j] = i * delta_x
        Y_grid[i, j] = j * delta_y

X_flat = X_grid.flatten()
Y_flat = Y_grid.flatten()

# 2. Setup Physics and Run RK4

print("Setting up simulation...")
advection_solver = CentralDifference3d(dx=delta_x, dy=delta_y, vel_x=v_x_array, vel_y=v_y_array)

# K_scat Matrix from paper
print("Loading pre-calculated ab initio data...")
ab_init_data = np.load('compiled_scat_rates_data.npz')
scattering_obj = two_state_transition_matrix(
    k_BB=ab_init_data['Rate_BB'],
    k_BD=ab_init_data['Rate_BD'],
    gamma_decay_constant=ab_init_data['radiative_rate'],
    map_Q_to_q=ab_init_data['Q_plus_q_map'],
    gamma_index=ab_init_data['gamma_index']
)

# Simulate for X femtoseconds
time_integrator = RungeKutta4(spatial_solver=advection_solver, total_sim_time=55.0, scattering_solver=scattering_obj)

# Run the integration and get the history of frames
print("Running RK4 Integration...")
frames = time_integrator.solve(occupation_matrix_3d, save_interval=2)

# 3. Visualization Loop
print("Simulation complete. Launching visualization...")

save_interval = 2

# q-grid
visualize_simulation(
    frames=frames,
    dt=time_integrator.dt,
    save_interval=save_interval,
    length_x=length_x,
    length_y=length_y,
    grid_x=grid_x,
    grid_y=grid_y,
    right_panel_mode='qgrid'
)
# q-slice
"""visualize_simulation(
    frames=frames,
    dt=time_integrator.dt,
    save_interval=save_interval,
    length_x=length_x,
    length_y=length_y,
    grid_x=grid_x,
    grid_y=grid_y,
    right_panel_mode='slice',
    target_state=13
)"""

# Post-Processing & Data Extraction
print("\nVisualization closed. Executing data extraction pipeline(s)...")

#export gif for presentation
export_diffusion_gif(
    frames=frames,
    dt=time_integrator.dt,
    save_interval=save_interval,
    length_x=length_x,
    length_y=length_y,
    grid_x=grid_x,
    grid_y=grid_y,
    right_panel_mode='qgrid',
    q_Nx=Nx,
    q_Ny=Ny,
    filename="diffusion_panels_0.05_300.gif"
)

extracted_D = extract_diffusion_constant(
    frames=frames,
    dt=time_integrator.dt,
    save_interval=save_interval,
    x_grid=X_grid,
    y_grid=Y_grid,
    cutoff_fraction=0.6,
    show_plot=True
)