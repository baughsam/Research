import h5py

xctph_h5 = "../../xctph_8x8x8.h5"

prefix = "8x8x8"
log_file_Qpts = f'{prefix}_Qpts'
log_file_qpts = f'{prefix}_qpts'
log_file_freq = f'{prefix}_freq'
log_file_energies = f'{prefix}_energies'

state_1 = 0
state_2 = 1
Q_state =4
q_state = 0
phonon_mode = 4


with h5py.File(xctph_h5, mode='r') as f:
    g_tensor = f['xctph'][:]
    Qpts = f['Qpts'][:]
    qpts = f['qpts'][:]
    freq = f['frequencies'][:]
    energies =  f['energies'][:]


num = g_tensor[state_1, state_2, Q_state, phonon_mode, q_state]
print(f"xctph scattering matrix elements at ({state_1}, {state_2}, {Q_state}, {phonon_mode}, {q_state}): {num}")

with open(log_file_Qpts, "w") as log:
    log.write("Qpts\n")
    for item in Qpts:
        log.write(f"{item}\n")
with open(log_file_qpts, "w") as log:
    log.write("qpts\n")
    for item in qpts:
        log.write(f"{item}\n")
with open(log_file_freq, "w") as log:
    log.write("frequencies\n")
    for item in freq:
        log.write(f"{item}\n")
with open(log_file_energies, "w") as log:
    log.write("energies\n")
    for item in energies:
        log.write(f"{item}\n")