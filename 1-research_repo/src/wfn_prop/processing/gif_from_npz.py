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
    filename="exciton_dynamics.gif")