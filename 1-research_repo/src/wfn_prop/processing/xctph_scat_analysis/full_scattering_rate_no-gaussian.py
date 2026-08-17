import h5py
import numpy as np
import scipy.constants as const
from datetime import datetime

# --- CONSTANTS & PARAMETERS ---
RY_TO_EV = 13.605698
HBAR_EV_FS = (const.hbar / const.e) * 1e15
GOLDEN_RULE_PREFACTOR = (2.0 * np.pi) / HBAR_EV_FS

temp_K = 300
sigma_eV = 0.02
bright_exciton_state = 0
dark_exciton_state = 1

S1 = f"S_{str(bright_exciton_state)}"
S2 = f"S_{str(dark_exciton_state)}"

xctph_h5 = "xctph_4x4x4.h5"
log_file = "test1_scattering_rate_4x4x4_no-gaussian.log"


# --- CORE PHYSICS FUNCTIONS ---
def bose_einstein_dist_2D(temp_K: float, freq_2D: np.ndarray) -> np.ndarray:
    k_B = 8.617333262e-5
    k_B_T = k_B * temp_K
    BED_2D = np.zeros_like(freq_2D)
    if temp_K == 0:
        return BED_2D
    valid_modes = freq_2D > 1e-6
    BED_2D[valid_modes] = 1.0 / (np.exp(freq_2D[valid_modes] / k_B_T) - 1.0)
    return BED_2D


def energy_gap_2D(Q_plus_q_map: np.ndarray, energies: np.ndarray, S_initial: int, S_final: int) -> np.ndarray:
    E_initial_1D = energies[S_initial, :]
    E_final_2D = energies[S_final, Q_plus_q_map]
    E_initial_2D = E_initial_1D[:, np.newaxis]
    return E_final_2D - E_initial_2D


def gaussian_weight(delta_E_2D: np.ndarray, frequencies_2D: np.ndarray, sigma_eV: float, process: str) -> np.ndarray:
    delta_E_3D = delta_E_2D[:, np.newaxis, :]
    hw_3D = frequencies_2D[np.newaxis, :, :]

    if process == 'absorption':
        x_3D = delta_E_3D - hw_3D
    elif process == 'emission':
        x_3D = delta_E_3D + hw_3D

    prefactor = 1.0 / (sigma_eV * np.sqrt(2.0 * np.pi))
    exponent = -0.5 * (x_3D / sigma_eV) ** 2
    return prefactor * np.exp(exponent)


def energy_conservation_term(BED_2D: np.ndarray, rho_abs_3D: np.ndarray, rho_emiss_3D: np.ndarray) -> np.ndarray:
    BED_3D = BED_2D[np.newaxis, :, :]
    absorption = BED_3D * rho_abs_3D
    emission = (BED_3D + 1) * rho_emiss_3D
    return absorption + emission


def compute_scattering_rates_no_gaussian(
        g_tensor_5D, energies, frequencies, Q_plus_q_map,
        S_initial, S_final, temp_K, sigma_eV, N_Q, N_q):
    """
    Computes both the total macroscopic scattering rate and
    the exciton-momentum (Q) resolved rates.
    """
    tensor_3D = g_tensor_5D[S_initial, S_final, :, :, :]
    tensor_3D_squared = np.abs(tensor_3D) ** 2

    BED_2D = bose_einstein_dist_2D(temp_K, frequencies)
    EG_2D = energy_gap_2D(Q_plus_q_map, energies, S_initial, S_final)

    rho_abs = gaussian_weight(EG_2D, frequencies, sigma_eV, 'absorption')
    rho_emi = gaussian_weight(EG_2D, frequencies, sigma_eV, 'emission')

    energy_cons_term = 1 #energy_conservation_term(BED_2D, rho_abs, rho_emi)
    weight_scat_rate_nu_Q_q = tensor_3D_squared * energy_cons_term

    # Integrate out all dimensions (Q, nu, q) for the total scalar rate
    total_rate = np.sum(weight_scat_rate_nu_Q_q)
    total_rate_fs = (total_rate / (N_Q * N_q)) * GOLDEN_RULE_PREFACTOR

    # Integrate out nu and q (axis 1 and 2) to get the Q-resolved array
    rate_Q = np.sum(weight_scat_rate_nu_Q_q, axis=(1, 2))
    rate_Q_fs = (rate_Q / N_q) * GOLDEN_RULE_PREFACTOR

    return total_rate_fs, rate_Q_fs


# --- MAIN EXECUTION BLOCK ---
if __name__ == "__main__":
    print(f"Loading data from {xctph_h5}...")
    with h5py.File(xctph_h5, mode='r') as f:
        g_tensor = f['xctph'][:] * RY_TO_EV
        energies = f['energies'][:] * RY_TO_EV
        frequencies = f['frequencies'][:] * RY_TO_EV
        Q_plus_q_map = f['Q_plus_q_map'][:]
        N_Q = f['nQ'][()]
        N_q = f['nq'][()]
        Qpts = f['Qpts'][:]

    # Locate the exact index for Q = (0,0,0)
    gamma_idx = np.where((Qpts[:, 0] == 0) & (Qpts[:, 1] == 0) & (Qpts[:, 2] == 0))[0][0]

    print("Computing Intraband Rates (S_B -> S_B)...")
    rate_total_BB, rate_Q_array_BB = compute_scattering_rates_no_gaussian(
        g_tensor, energies, frequencies, Q_plus_q_map,
        bright_exciton_state, bright_exciton_state, temp_K, sigma_eV, N_Q, N_q
    )
    time_total_BB = 1.0 / rate_total_BB if rate_total_BB > 0 else np.nan
    rate_gamma_BB = rate_Q_array_BB[gamma_idx]
    time_gamma_BB = 1.0 / rate_gamma_BB if rate_gamma_BB > 1e-12 else np.nan

    print("Computing Interband Rates (S_B -> S_D)...")
    rate_total_BD, rate_Q_array_BD = compute_scattering_rates_no_gaussian(
        g_tensor, energies, frequencies, Q_plus_q_map,
        bright_exciton_state, dark_exciton_state, temp_K, sigma_eV, N_Q, N_q
    )
    time_total_BD = 1.0 / rate_total_BD if rate_total_BD > 0 else np.nan
    rate_gamma_BD = rate_Q_array_BD[gamma_idx]
    time_gamma_BD = 1.0 / rate_gamma_BD if rate_gamma_BD > 1e-12 else np.nan

    # --- WRITE TO LOG FILE ---
    print(f"Writing results to {log_file}...")
    with open(log_file, "w") as log:
        log.write("=" * 50 + "\n")
        log.write("EXCITON-PHONON SCATTERING RATE COMPUTATION LOG (NO GAUSSIAN)\n")
        log.write("=" * 50 + "\n")
        log.write(f"Run Date/Time     : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        log.write(f"Input Data File   : {xctph_h5}\n\n")

        log.write("--- PARAMETERS ---\n")
        log.write(f"Temperature       : {temp_K} K\n")
        #log.write(f"Broadening (sigma): {sigma_eV} eV\n")
        log.write(f"Bright State (SB) : Index {bright_exciton_state}\n")
        log.write(f"Dark State (SD)   : Index {dark_exciton_state}\n")
        log.write(f"Q-Grid Size (N_Q) : {N_Q}\n")
        log.write(f"q-Grid Size (N_q) : {N_q}\n\n")

        log.write("--- TOTAL SCATTERING RESULTS (GRID INTEGRATED) ---\n")
        log.write(f"Intraband ({S1} -> {S1}) Total Rate : {rate_total_BB:.6e} fs^-1\n")
        log.write(f"Intraband ({S1} -> {S1}) Total Time : {time_total_BB:.2f} fs\n\n")
        log.write(f"Interband ({S1} -> {S2}) Total Rate : {rate_total_BD:.6e} fs^-1\n")
        log.write(f"Interband ({S1} -> {S2}) Total Time : {time_total_BD:.2f} fs\n\n")

        log.write("--- GAMMA POINT RESULTS (Q = 0,0,0) ---\n")
        log.write(f"Intraband ({S1} -> {S1}) Gamma Rate : {rate_gamma_BB:.6e} fs^-1\n")
        log.write(f"Intraband ({S1} -> {S1}) Gamma Time : {time_gamma_BB:.2f} fs\n\n")
        log.write(f"Interband ({S1} -> {S2}) Gamma Rate : {rate_gamma_BD:.6e} fs^-1\n")
        log.write(f"Interband ({S1} -> {S2}) Gamma Time : {time_gamma_BD:.2f} fs\n")
        log.write("=" * 50 + "\n")

    print(f"Analysis complete. See {log_file} for details.")