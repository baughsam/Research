import numpy as np
import h5py
from wfn_prop.analysis import export_diffusion_gif_updated1, export_diffusion_gif_updated2

# 1. Open the HDF5 file in read-only mode
h5_filename = "../LiF_8x8x8_Kscat-on_intraband-only_spread_initialization_S0_200fs_frames.h5"
h5_file = h5py.File(h5_filename, 'r')

# 2. Extract the frames dataset pointer (This does NOT load it into RAM)
frames = h5_file['frames']

dt = h5_file.attrs['dt']

# 3. Load the original momentum state vectors
physics = np.load("../compiled_scat_rates_data_S0_S1_8x8x8.npz")
q_vectors = physics['Qpts']

# 4. Generate the two-panel animated GIF
export_diffusion_gif_updated2(
    frames=frames,
    dt=dt,
    save_interval=2,
    length_x=100,
    length_y=100,
    grid_x=250,
    grid_y=250,
    right_panel_mode='qgrid',
    q_vectors=q_vectors,
    projection=('x', 'y'),
    filename="test2_LiF_8x8x8_Kscat-on_intraband-only_spread_initialization_S0_200fs.gif"
)

# 5. Cleanly close the file when the animation is finished baking
h5_file.close()