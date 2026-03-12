import numpy as np
import matplotlib.pyplot as plt
from wfn_prop.NumMthds import RungeKutta4, UpwindDifference2d

# Gaussian Distribution Function
def gaussian_dist_2d(x_pos, y_pos, spread_x, spread_y, amplitude, center_x, center_y):
    term1 = - ( (x_pos - center_x)**2 ) / (2 * spread_x**2)
    term2 = - ( (y_pos - center_y)**2 ) / (2 * spread_y**2)
    dist_funct = amplitude * np.exp( term1 + term2 )
    return dist_funct

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

# Initialize Grid
occupation_matrix = np.empty((grid_x, grid_y))

for i in range(grid_x):
    for j in range(grid_y):
        x_i = i * delta_x
        y_i = j * delta_y
        occupation_matrix[i,j] = gaussian_dist_2d(x_pos=x_i, y_pos=y_i, spread_x=sigma_x, spread_y=sigma_y, amplitude=amplitude, center_x=x_0, center_y=y_0)

# After occupation_matrix

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
# Give it some velocity so it moves!
advection_solver = UpwindDifference2d(dx=delta_x, dy=delta_y, velocity_x=0, velocity_y=3)

# Simulate for 15 femtoseconds
time_integrator = RungeKutta4(spatial_solver=advection_solver, total_sim_time=15.0)

# Run the integration and get the history of frames
print("Running RK4 Integration...")
frames = time_integrator.solve(occupation_matrix, save_interval=2)

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

    frame_flat = frame.flatten()

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