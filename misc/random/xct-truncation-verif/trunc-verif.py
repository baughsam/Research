import h5py
import numpy as np

output_filename = "test-norm-trunc-verif-6c_6v.out"

with h5py.File('xct.h5', 'r') as f:
    # Shape: (ns, nv, nc, nk, nevecs, nQ)
    avck = f['exciton_data/eigenvectors'][()]
    energies = f['exciton_data/eigenvalues'][()]

    nv_full = f['exciton_header/nv'][()]
    nc_full = f['exciton_header/nc'][()]
    nevecs_total = f['exciton_header/nevecs'][()]
    nQ = f['exciton_header/nQ'][()]

# Number of bands to truncate down to
nv_trunc = 6
nc_trunc = 6

#
n_full = nc_full * nv_full
n_trunc = nv_trunc * nc_trunc

# Only check the lowest states used in compute_xctph (e.g., nbnd_xct = 4)
n_states_to_check = 4

with open(output_filename, 'w') as f_out:
    f_out.write("--- Exciton Basis Truncation Verification ---\n")
    f_out.write(f"Original Basis:   {nv_full}v x {nc_full}c ({n_full} bands total)\n")
    f_out.write(f"Truncated Basis:  {nv_trunc}v x {nc_trunc}c ({n_trunc} bands total)\n")
    f_out.write(f"Total States in File: {nevecs_total} | Total Q-points: {nQ}\n\n")

    f_out.write(
        f"{'State (S)':>10} {'Q-idx':>8} {'Energy (eV)':>14} {'Total Norm':>12} {'Trunc Norm':>12} {'Retained %':>12}\n")
    f_out.write("-" * 72 + "\n")

    # Loop over the lowest exciton states and all Q-points
    for iS in range(n_states_to_check):
        for iQ in range(nQ):
            # Extract individual exciton state S at momentum Q
            full_wavepacket = avck[0, :, :, :, iS, iQ]

            # Slice top 6 valence (HOMO down) and bottom 6 conduction (LUMO up)
            truncated_wavepacket = avck[0, :nv_trunc, :nc_trunc, :, iS, iQ]

            total_norm = np.sum(np.abs(full_wavepacket) ** 2)
            truncated_norm = np.sum(np.abs(truncated_wavepacket) ** 2)
            retained_percentage = (truncated_norm / total_norm) * 100

            f_out.write(
                f"{iS:>10d} {iQ:>8d} {energies[iS, iQ]:>14.4f} "
                f"{total_norm:>12.4f} {truncated_norm:>12.4f} "
                f"{retained_percentage:>11.2f}%\n"
            )

print(f"Success. Data written to {output_filename}")