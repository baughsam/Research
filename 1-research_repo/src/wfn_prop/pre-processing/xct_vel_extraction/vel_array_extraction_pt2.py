import numpy as np
import scipy.constants as const

input_file = "ordered_raw_energies_state_1.npz"
output_file = "final_velocity_payload.npz"
HBAR_EV_FS = (const.hbar / const.e) * 1e15  # eV*fs

print(f"Loading raw energies from {input_file}...")
data = np.load(input_file)
Qpts = data['Qpts']
energies_1D = data['energies']

# 1. Determine Grid Dimensions
q_x_unique = np.unique(Qpts[:, 0])
q_y_unique = np.unique(Qpts[:, 1])
q_z_unique = np.unique(Qpts[:, 2])

Nx, Ny, Nz = len(q_x_unique), len(q_y_unique), len(q_z_unique)
dqx = q_x_unique[1] - q_x_unique[0] if Nx > 1 else 1.0
dqy = q_y_unique[1] - q_y_unique[0] if Ny > 1 else 1.0
dqz = q_z_unique[1] - q_z_unique[0] if Nz > 1 else 1.0

# 2. Map to 3D Tensor
E_3D = np.zeros((Nx, Ny, Nz))
mapping_indices = []

for q_idx, q_vec in enumerate(Qpts):
    i = np.where(q_x_unique == q_vec[0])[0][0]
    j = np.where(q_y_unique == q_vec[1])[0][0]
    k = np.where(q_z_unique == q_vec[2])[0][0]

    E_3D[i, j, k] = energies_1D[q_idx]
    mapping_indices.append((i, j, k))

# 3. Calculate 3D Velocity Gradient (With Periodic Boundaries)
print("Calculating velocity field with periodic boundary padding...")

# Wrap the edges of the Brillouin zone to enforce periodic boundaries
E_3D_padded = np.pad(E_3D, pad_width=1, mode='wrap')

# Calculate the gradient on the padded array
grad_E_padded = np.gradient(E_3D_padded, dqx, dqy, dqz)

# Slice off the artificial padding (index 1 to -1) to restore the original Nx, Ny, Nz dimensions,
# then apply the HBAR_EV_FS constant to convert to proper velocity units.
v_3D_x = grad_E_padded[0][1:-1, 1:-1, 1:-1] / HBAR_EV_FS
v_3D_y = grad_E_padded[1][1:-1, 1:-1, 1:-1] / HBAR_EV_FS
v_3D_z = grad_E_padded[2][1:-1, 1:-1, 1:-1] / HBAR_EV_FS

# 4. Flatten back to 1D arrays using the original order
v_x_1D, v_y_1D, v_z_1D = np.zeros(len(Qpts)), np.zeros(len(Qpts)), np.zeros(len(Qpts))

for q_idx, (i, j, k) in enumerate(mapping_indices):
    v_x_1D[q_idx] = v_3D_x[i, j, k]
    v_y_1D[q_idx] = v_3D_y[i, j, k]
    v_z_1D[q_idx] = v_3D_z[i, j, k]

np.savez(output_file, vel_x=v_x_1D, vel_y=v_y_1D, vel_z=v_z_1D, energy=energies_1D, Qpts=Qpts)
print(f"SUCCESS: Velocities calculated and saved to {output_file}.")