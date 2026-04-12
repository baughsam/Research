import numpy as np
import matplotlib.pyplot as plt
from wfn_prop.NumMthds import RungeKutta4, UpwindDifference2d, UpwindDifference3d
from wfn_prop.k_scat import Decay, FickDiff, PhononScat

# Gaussian Distribution Function
def gaussian_dist_2d(x_pos, y_pos, spread_x, spread_y, amplitude, center_x, center_y):
    term1 = - ( (x_pos - center_x)**2 ) / (2 * spread_x**2)
    term2 = - ( (y_pos - center_y)**2 ) / (2 * spread_y**2)
    dist_funct = amplitude * np.exp( term1 + term2 )
    return dist_funct

def initialize_tranisiton_matrix(energies_eV: np.ndarray, temp_K: float, coupling_constant: float) -> np.ndarray:
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

    #Bose-Einstein Statistics
    with np.errstate(divide='ignore', invalid='ignore'):
        N_phonons = 1.0 /  (np.exp(np.abs(dE) / k_B_T) - 1.0)

    #Initialize Scattering Matrix (empty)
    W = np.zeros((Nq, Nq))

    # Fermi's Golden Rule (Absorption and Emission)
    # Upward Transition (Absorption)
    W[dE > 0] = coupling_constant * N_phonons[dE > 0]
    # Downward Transition (Emission)
    W[dE < 0] = coupling_constant * N_phonons[dE < 0] + 1 # +1 = Spontaneous Emission due to  Heisenburg Uncetainty Principle (delta_E*delta_t >= hbar/2)

    #Particle Conversation (Dealing w/ the Diagonal)
    # - Excitons cannot scatter into their own states
    np.fill_diagonal(W, 0.0)
    # - Diagonal W[i,i] must represent the total rate of excitons leaving state i
    drain_rate = -np.sum(W, axis=1) # Sums everything across a row # np.ndarray
    np.fill_diagonal(W, drain_rate)

    return W

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

    return E_1D


# Initializing Gaussian Wavepack on N_x x N_y sized grid
# Real Space Dimension in nanometers
length_x = 40
length_y = 40

# Grid Dimensions
grid_x = 100
grid_y = 100

# Simulation Distances
delta_x = length_x / (grid_x-1)
delta_y = length_y / (grid_y-1)

#Box (real space) center
x_0 = length_x / 2
y_0 = length_y / 2

# Gaussian Spread (Change to actual simulation values)
sigma_x = 5
sigma_y = 5

# Amplitude
amplitude = 1

# Initialize 2D Grid
occupation_matrix_2d = np.empty((grid_x, grid_y))

# Momentum Space (currently an arbitrary 3x3 dimensional array)
Nq = 3

# Velocity Arrays (would be calculated form the exciton dispersion)
v_x_array = np.array([-1.0, 0.0, 1.0])
v_y_array = np.array([0.0, 0.0, 0.0])

# Initialize 3D array
occupation_matrix_3d = np.empty((grid_x, grid_y, Nq))

for i in range(grid_x):
    for j in range(grid_y):
        x_i = i * delta_x
        y_i = j * delta_y
        occupation_matrix_3d[i,j, 2] = gaussian_dist_2d(x_pos=x_i, y_pos=y_i, spread_x=sigma_x, spread_y=sigma_y, amplitude=amplitude, center_x=x_0, center_y=y_0)



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
#Choose K_scat
#scattering_obj = Decay(scat_time=15)

print("Setting up simulation...")
advection_solver = UpwindDifference3d(dx=delta_x, dy=delta_y, vel_x=v_x_array, vel_y=v_y_array) #UpwindDifference2d(dx=delta_x, dy=delta_y, velocity_x=1, velocity_y=1)

# Fake a scattering matrix where State 2 (Right) decays into State 1 (Stationary)
W_matrix = np.zeros((Nq, Nq))
W_matrix[2, 2] = -0.05  # Losing excitons from State 2
W_matrix[2, 1] = +0.05  # Gaining them in State 1
scattering_obj = PhononScat(transition_matrix=W_matrix)

# Simulate for 15 femtoseconds
time_integrator = RungeKutta4(spatial_solver=advection_solver, total_sim_time=40.0, scattering_solver=scattering_obj)

# Run the integration and get the history of frames
print("Running RK4 Integration...")
frames = time_integrator.solve(occupation_matrix_3d, save_interval=2)

# 3. Visualization Loop
print("Simulation complete. Launching visualization...")

save_interval = 2
physical_time_per_frame = time_integrator.dt * save_interval

plt.ion()
# Made the figure slightly wider (7, 6) to comfortably fit the new colorbar
fig, ax = plt.subplots(figsize=(7, 6))

colorbar_created = False

for i, frame in enumerate(frames):
    ax.clear()
    # 3D Tensor Addition
    # Sum accross the Q-axis (axis=2) to get the physical 2D density map
    physical_density = np.sum(frame, axis=2)

    #frame_flat = physical_density.flatten()

    # Slices the 3D tensor to ONLY look at State 2 (Moving Right)
    q_slice_density = frame[:, :, 1]
    frame_flat = q_slice_density.flatten()

    # We add vmin=0.0 and vmax=amplitude to strictly lock the physics color scale
    sc = ax.scatter(X_flat, Y_flat, c=frame_flat, cmap='magma', marker='s', s=10, vmin=0.0, vmax=amplitude)

    # Draw the colorbar ONLY on the very first frame
    if not colorbar_created:
        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label("Exciton Density")
        colorbar_created = True

    current_time_fs = i * physical_time_per_frame

    ax.set_title(f"Ballistic Transport | Time: {current_time_fs:.3f} fs")
    ax.set_xlabel("X Position (nm)")
    ax.set_ylabel("Y Position (nm)")
    ax.set_xlim(0, length_x)
    ax.set_ylim(0, length_y)

    plt.draw()
    plt.pause(0.05)

plt.ioff()
plt.show()