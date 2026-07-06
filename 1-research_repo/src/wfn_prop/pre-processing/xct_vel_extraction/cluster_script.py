import os
import sys
import numpy as np
import h5py

target_state = 1  # Exciton state to extract
xctph_h5_path = "xctph.h5"
bands_in_path = "bands.in"
output_file = f"ordered_raw_energies_state_{target_state}.npz"


def build_q_to_folder_map(bands_filename):
    """Maps coordinates to QXXXX directories from bands.in."""
    q_map = {}
    with open(bands_filename, 'r') as f:
        lines = f.readlines()

    start_idx = next(i for i, line in enumerate(lines) if "K_POINTS" in line) + 2
    for counter, line in enumerate(lines[start_idx:]):
        if line.strip():
            parts = line.split()
            qx, qy, qz = round(float(parts[0]), 5), round(float(parts[1]), 5), round(float(parts[2]), 5)
            q_map[(qx, qy, qz)] = f"Q{counter:04d}"

    return q_map


print("Reading master Qpts array from xctph.h5...")
with h5py.File(xctph_h5_path, 'r') as f:
    master_Qpts = f['Qpts'][:]
N_Q = len(master_Qpts)

folder_map = build_q_to_folder_map(bands_in_path)
ordered_energies = np.zeros(N_Q)

print("Extracting energies in master order...")
for q_idx, q_vec in enumerate(master_Qpts):
    q_tuple = (round(q_vec[0], 5), round(q_vec[1], 5), round(q_vec[2], 5))
    folder = folder_map[q_tuple]
    file_path = os.path.join(folder, '03-singlet', 'eigenvalues_b1.dat')

    # --- STRICT FAILSAFE ---
    if not os.path.exists(file_path):
        print(f"\nFATAL ERROR: Missing file {file_path}")
        print("Extraction halted. Ensure all cluster array jobs have finished successfully.")
        sys.exit(1)

    with open(file_path, 'r') as f:
        valid_lines = [line for line in f if not line.startswith('#') and line.strip()]

        # Secondary failsafe: Check if the file exists but crashed before writing the target state
        if target_state - 1 >= len(valid_lines):
            print(f"\nFATAL ERROR: {file_path} is incomplete. Target state {target_state} not found.")
            print("Extraction halted. Check the absorption.log file in this directory.")
            sys.exit(1)

        ordered_energies[q_idx] = float(valid_lines[target_state - 1].split()[0])

np.savez(output_file, Qpts=master_Qpts, energies=ordered_energies)
print(f"SUCCESS: Exported {output_file}. Ready to download to local machine.")