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




# Initializing Gaussian Wavepack
# From: Signatures of Dimensionality and Symmetry in Exciton Band
#       Structure: Consequences for Exciton Dynamics and Transport

# xctph .npz file
xctph_npz = 'compiled_scat_rates_data.npz'

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


# 1. Recreate the X and Y grids for plotting (matching your loop logic)
X_grid = np.empty((grid_x, grid_y))
Y_grid = np.empty((grid_x, grid_y))
for i in range(grid_x):
    for j in range(grid_y):
        X_grid[i, j] = i * delta_x
        Y_grid[i, j] = j * delta_y

X_flat = X_grid.flatten()
Y_flat = Y_grid.flatten()

# Gaussian Wavefunction Initialization (Q-Space)

# Load pre-compiled physics .npz file (from compile_scat_rates.py)
print("Loading pre-calculated ab initio .npz file...")
physics_payload = np.load(xctph_npz)

Q_vectors = physics_payload['qpoints']
N_Q = len(Q_vectors)

# GROUP VELOCITY PLACEHOLDER
v_x_array = np.zeros(N_Q)
v_y_array = np.zeros(N_Q)

# Initializing Gaussian Wavepack
# From: Signatures of Dimensionality and Symmetry in Exciton Band
#       Structure: Consequences for Exciton Dynamics and Transport
sigma_R = 3.0 # nm
sigma_Q = 2.0 # nm^-1
amplitude = 1

print("Initializing bright exciton phase-space wavepacket...")
occupation_matrix_3d = np.zeros((grid_x,grid_y, N_Q))

for q_index in range(N_Q):
    Q_vector = Q_vectors[q_index]
    magnitude_Q_sq = np.sum(Q_vector**2)
    weight_Q = np.exp(-magnitude_Q_sq / (2 * sigma_Q**2))

    spatial_dist = gaussian_dist_2d(X_grid, Y_grid, spread_x=sigma_R, spread_y=sigma_R,
                                    amplitude=amplitude, center_x=x_0, center_y=y_0)

    occupation_matrix_3d[:, :, q_index] = spatial_dist * weight_Q

# 2. Setup Physics and Run RK4

print("Setting up simulation...")
advection_solver = CentralDifference3d(dx=delta_x, dy=delta_y, vel_x=v_x_array, vel_y=v_y_array)

# K_scat Matrix from paper
print("Initializing K_scat object...")
ab_init_data = physics_payload
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
    q_Nx=N_Q,
    q_Ny=N_Q,
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