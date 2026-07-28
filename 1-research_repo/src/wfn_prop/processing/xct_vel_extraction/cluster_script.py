import os
import sys
import numpy as np
import h5py

target_state = 1  # Exciton state to extract
xctph_h5_path = "xctph.h5"
output_file = f"ordered_raw_energies_state_{target_state}.npz"

print("Reading master Qpts array from xctph.h5...")
with h5py.File(xctph_h5_path, 'r') as f:
    master_Qpts = f['Qpts'][:]
N_Q = len(master_Qpts)

# Extract the reciprocal lattice matrix
print("Reading reciprocal lattice from eph.h5...")
with h5py.File(eph_h5_path, 'r') as f:
    recip_lat_bohr = f['gkq_header/recip_lat'][()]

ordered_energies = np.zeros(N_Q)

print(f"Extracting energies for {N_Q} Q-points based on alphabetical glob ordering...")
for q_idx in range(N_Q):

    # Because write_xct_h5.py uses np.sort(glob.glob(...)),
    # the index in the h5 file maps 1:1 to the Q directory number!
    folder = f"Q{q_idx:04d}"
    file_path = os.path.join(folder, '03-singlet', 'eigenvalues_b1.dat')

    # --- STRICT FAILSAFE ---
    if not os.path.exists(file_path):
        print(f"\nFATAL ERROR: Missing file {file_path}")
        print("Extraction halted. Ensure all cluster array jobs have finished successfully.")
        sys.exit(1)

    with open(file_path, 'r') as f:
        valid_lines = [line for line in f if not line.startswith('#') and line.strip()]

        # Secondary failsafe for incomplete writes
        if target_state - 1 >= len(valid_lines):
            print(f"\nFATAL ERROR: {file_path} is incomplete. Target state {target_state} not found.")
            print("Extraction halted. Check the absorption.log file in this directory.")
            sys.exit(1)

        ordered_energies[q_idx] = float(valid_lines[target_state - 1].split()[0])

np.savez(output_file, Qpts=master_Qpts, energies=ordered_energies)
print(f"SUCCESS: Exported {output_file}. Ready to download to local machine.")