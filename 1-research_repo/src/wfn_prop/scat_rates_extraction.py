import h5py
import scipy.constants as const
import numpy as np

xctph_h5 = "xctph.h5"
bright_exciton_state = 0
dark_exciton_state = 1

RY_TO_EV = 13.605698

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




