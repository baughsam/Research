import h5py
import scipy.constants as const
import numpy as np

xctph_h5 = "xctph.h5"
bright_exciton_state = 0
dark_exciton_state = 1
temp_K = 300
sigma_eV = 0.02

RY_TO_EV = 13.605698
HBAR_EV_FS = (const.hbar / const.e) * 1e15
GOLDEN_RULE_PREFACTOR = (2.0 * np.pi) / HBAR_EV_FS

# Open HDF5 file in read-only mode
with h5py.File(xctph_h5, mode = 'r') as f:

    # Load datasets into NumPy arrays in memory
    # (Might be an issue later when our 5D g_tensor is huge)
    g_tensor = f['xctph'][:]                       # Shape: (S, S', Q, nu, q)
    energies = f['energies'][:] * RY_TO_EV         # Shape: (S, Q)
    frequencies = f['frequencies'][:] * RY_TO_EV   # Shape: (nu, q)
    Q_plus_q_map = f['Q_plus_q_map'][:]            # Shape: (Q, q)

# NOTE: Data remains in NumPy arrays after exiting the 'with' block

# Slice exciton-phonon scattering matrix elements down to 3D
tensor_BB_3D = g_tensor[bright_exciton_state, bright_exciton_state, :, :, :] # Intraband
tensor_BD_3D = g_tensor[bright_exciton_state, dark_exciton_state, :, :, :]   # Interband

# Calculate the weighting tensor

def bose_einstein_dist_2D(temp_K: float, freq_2D: np.ndarray) -> np.ndarray:
    """
    :param temp_K: Temperature in Kelvin
    :param freq_2D: 2D frequency array Shape: (nu, q)
    :return: 2D Bose-Einstein distribution as it changes across phonon mode and phonon momentum
    """
    k_B = 8.617333262e-5 #eV/K
    k_B_T = k_B * temp_K

    if temp_K == 0:
        BED_2D = np.zeros_like(freq_2D)
        print("Temperature is 0K. Setting Bose-Einstein distribution to zero.")

    BED_2D = np.zeros_like(freq_2D)

    # Create mask to only calculate where the frequency is greater than 0
    # Avoids 1.0 / 0.0 acousted mode failure
    # (1e-6 ws chosen because it is a generally good value)
    valid_modes = freq_2D > 1e-6

    BED_2D[valid_modes] = 1.0 / (np.exp(freq_2D[valid_modes] / k_B_T) - 1.0)

    return BED_2D

def energy_gap_2D(Q_plus_q_map: np.ndarray, energies: np.ndarray, S_initial: int, S_final: int) -> np.ndarray:
    """
    Calculates the energy gap 2D aray for a specific band transition

    :param Q_plus_q_map: 2D array of mapped target indices, Shape: (Q, q)
    :param energies: 2D array of exciton energies, Shape: (S, Q)
    :param S_initial: Integer index of the initial state
    :param S_final: Integer index of the final state
    :return: 2D array of Delta E values, Sheap: (Q, q)
    """

    # 1D array of initial energies, Shape: (N_Q,)
    E_initial_1D = energies[S_initial, :]

    # 2D array of final energies using our mapping
    # Q_plus_q_map is an (N_Q, N_q) sized array; passing into 1D array gives, Shape: (N_Q, N_q)
    E_final_2D = energies[S_final, Q_plus_q_map]

    # Stretch the inital energies to 2D for subtration, Shape: (N_Q, 1)
    E_initial_2D = E_initial_1D[:, np.newaxis]

    # Calculate the gap, Shape: (N_Q, N_q)
    delta_E = E_final_2D - E_initial_2D

    return delta_E

def gaussian_weight(delta_E_2D: np.ndarray, frequencies_2D: np.ndarray, sigma_eV: float, process: str = 'absorption') -> np.ndarray:
    """
    Calculates the 3D Gaussian broadening tensor for a specific scattering process
    :param delta_E_2D: 2D energy gap array, Shape: (Q, q)
    :param frequencies_2D: 2D phonon frequency array, Shape: (nu, q)
    :param sigma_eV: Gaussian broadening parameter in eV
    :param process: 'absorption' or 'emission'
    :return:  3D Gaussian weights, Shape: (Q, nu, q)
    """

    #delta_E_2D is (Q, q). In needs to be (Q, nu, q)
    delta_E_3D = delta_E_2D[:, np.newaxis, :] # Shape (Q, 1, q)

    # freq_2D is (nu, q). It needs to be (Q, nu, q)
    hw_3D = frequencies_2D[np.newaxis, :, :] # Shape: (1, nu, q)

    #These new 3D tensors will interact and the resulting 3D tensor will have the correct size (Q, nu, q)
    if process == 'absorption':
        x_3D = delta_E_3D - hw_3D
    elif process == 'emission':
        x_3D = delta_E_3D + hw_3D
    else:
        raise ValueError("Process must be 'absorption' or 'emission'")

    # Gaussian Operations (apply to all elements in 3D tensor)
    prefactor = 1.0 / (sigma_eV * np.sqrt(2.0 * np.pi))
    exponent = -0.5 * (x_3D / sigma_eV)**2

    rho_3D = prefactor * np.exp(exponent)

    return rho_3D

def energy_conservation_term(BED_2D: np.ndarray, rho_abs_3D: np.ndarray, rho_emiss_3D: np.ndarray) -> np.ndarray:
    """
    Combines the thermodynamic (Bose-Einstein) and kinematic (Gaussian)
    factors into a single 3D phase-space tensor for scattering
    :param BED_2D: 2D Bose-Enistein distribution, Shape: (nu, q)
    :param rho_abs_3D: 3D Gaussian broadening for absorption, Shape: (Q, nu, q)
    :param rho_emiss_3D: 3D Gaussian broadening for emission, Shape: (Q, nu, q)
    :return: 3D energy conservation weighting tensor
    """

    # Stretch 2D Bose-Einstein array to 3D
    BED_3D = BED_2D[np.newaxis, :, :]

    # Absorption
    absorption = BED_3D * rho_abs_3D

    # Emission
    emission = (BED_3D + 1) * rho_emiss_3D

    full_energy_conservation_term = absorption + emission

    return full_energy_conservation_term

def compute_transition_rates(
        g_tensor_5D: np.ndarray,
        energies: np.ndarray,
        frequencies: np.ndarray,
        Q_plus_q_map: np.ndarray,
        S_initial: int,
        S_final: int,
        temp_K: float,
        sigma_eV: float
) -> np.ndarray:
    """
    Extracts and computes the flattened 2D scattering rate matrix for a specific
    exciton state transition by evaluating phase-space conservation and
    integrating out the phonon modes.
    :return: 2D array of scattering rates. Shape: (Q, q)
    """

    # 1. Slice and square the tensor for the specific transition
    tensor_3D = g_tensor_5D[S_initial, S_final, :, :, :]
    tensor_3D_squared = np.abs(tensor_3D) ** 2

    # 2. Thermodynamic weighting
    BED_2D = bose_einstein_dist_2D(temp_K=temp_K, freq_2D=frequencies)

    # 3. Kinematic mapping (State-dependent)
    EG_2D = energy_gap_2D(
        Q_plus_q_map=Q_plus_q_map,
        energies=energies,
        S_initial=S_initial,
        S_final=S_final
    )

    # 4. Phase space broadening
    rho_abs = gaussian_weight(
        delta_E_2D=EG_2D, frequencies_2D=frequencies,
        sigma_eV=sigma_eV, process='absorption'
    )
    rho_emi = gaussian_weight(
        delta_E_2D=EG_2D, frequencies_2D=frequencies,
        sigma_eV=sigma_eV, process='emission'
    )

    # 5. Total energy conservation term
    energy_cons_term = energy_conservation_term(
        BED_2D=BED_2D, rho_abs_3D=rho_abs, rho_emiss_3D=rho_emi
    )

    # 6. Apply weights and integrate out the phonon modes
    weight_scat_rate_nu_Q_q = tensor_3D_squared * energy_cons_term
    weight_scat_rate_Q_q = np.sum(weight_scat_rate_nu_Q_q, axis=1) # Units: eV

    # Convert energy linewidth to scattering rates (Units: 1/fs)
    weight_scat_rate_Q_q_fs = weight_scat_rate_Q_q * GOLDEN_RULE_PREFACTOR

    return weight_scat_rate_Q_q_fs

# --- Compute Intraband Matrix (B -> B) ---
Rate_BB = compute_transition_rates(
    g_tensor_5D=g_tensor,
    energies=energies,
    frequencies=frequencies,
    Q_plus_q_map=Q_plus_q_map,
    S_initial=bright_exciton_state,
    S_final=bright_exciton_state,
    temp_K=temp_K,
    sigma_eV=sigma_eV
)

# --- Compute Interband Matrix (B -> D) ---
Rate_BD = compute_transition_rates(
    g_tensor_5D=g_tensor,
    energies=energies,
    frequencies=frequencies,
    Q_plus_q_map=Q_plus_q_map,
    S_initial=bright_exciton_state,
    S_final=dark_exciton_state,
    temp_K=temp_K,
    sigma_eV=sigma_eV
)

# Find the Gamma point index dynamically (for the placement of the radiative decay rate in K_scat matrix)
# We re-open the h5 file briefly just to grab the qpoints if they exist
with h5py.File(xctph_h5, mode='r') as f:
    try:
        Q_vectors = f['Qpts'][:]
        is_gamma = np.all(np.isclose(Q_vectors, 0.0), axis=1)
        gamma_index = int(np.where(is_gamma)[0][0])
        print(f"Gamma point located at index: {gamma_index}")
    except KeyError:
        print("WARNING: 'Qpts' key not found in h5. Defaulting Gamma index to 0.")

# Radiative Decay (Bright @ Gamma -> Ground)
# Calculate this from the dipole strength as per Term 4 in the paper
radiative_rate_fs = 0.005 #

# 6. Export the Payload
output_filename = 'compiled_scat_rates_data.npz'
np.savez(
    output_filename,
    Rate_BB=Rate_BB,
    Rate_BD=Rate_BD,
    Q_plus_q_map=Q_plus_q_map,
    gamma_index=gamma_index,
    radiative_rate=radiative_rate_fs
)

print(f"\nSUCCESS: Exported fully coupled physics payload to {output_filename}")
