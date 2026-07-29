import h5py
import numpy as np
import scipy.constants as const
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# --- CONSTANTS & PARAMETERS ---
RY_TO_EV = 13.605698
HBAR_EV_FS = (const.hbar / const.e) * 1e15
GOLDEN_RULE_PREFACTOR = (2.0 * np.pi) / HBAR_EV_FS
BOHR_TO_NM = const.physical_constants['Bohr radius'][0] * 1e9

temp_K = 300
sigma_eV = 0.02
bright_exciton_state = 0
dark_exciton_state = 1

S1 = f"S_{str(bright_exciton_state)}"
S2 = f"S_{str(dark_exciton_state)}"

# Target your local xctph file
# (Matches the 4x4x4 grid from your corrections)
xctph_h5 = "xctph_4x4x4.h5"
raw_energies_file = "../xct_vel_extraction/ordered_raw_energies_state_1_4x4x4.npz"


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


def compute_mode_resolved_rates(
        g_tensor_5D, energies, frequencies, Q_plus_q_map,
        S_initial, S_final, temp_K, sigma_eV, N_Q, N_q):
    """
    Computes scattering rates resolved per phonon mode.
    """
    tensor_3D = g_tensor_5D[S_initial, S_final, :, :, :]
    tensor_3D_squared = np.abs(tensor_3D) ** 2

    BED_2D = bose_einstein_dist_2D(temp_K, frequencies)
    EG_2D = energy_gap_2D(Q_plus_q_map, energies, S_initial, S_final)

    rho_abs = gaussian_weight(EG_2D, frequencies, sigma_eV, 'absorption')
    rho_emi = gaussian_weight(EG_2D, frequencies, sigma_eV, 'emission')

    energy_cons_term = energy_conservation_term(BED_2D, rho_abs, rho_emi)
    weight_scat_rate_nu_Q_q = tensor_3D_squared * energy_cons_term

    # Integrate out Q and q to isolate the mode (nu)
    rate_nu = np.sum(weight_scat_rate_nu_Q_q, axis=(0, 2))
    rate_nu_fs = (rate_nu / (N_Q * N_q)) * GOLDEN_RULE_PREFACTOR

    # Average phonon energy over q for the x-axis
    avg_phonon_energy_eV = np.mean(frequencies, axis=1)
    return avg_phonon_energy_eV, rate_nu_fs


def compute_momentum_resolved_rates(
        g_tensor_5D, energies, frequencies, Q_plus_q_map,
        S_initial, S_final, temp_K, sigma_eV, N_Q, qpts):
    """
    Computes scattering rates resolved per phonon momentum q.
    """
    tensor_3D = g_tensor_5D[S_initial, S_final, :, :, :]
    tensor_3D_squared = np.abs(tensor_3D) ** 2

    BED_2D = bose_einstein_dist_2D(temp_K, frequencies)
    EG_2D = energy_gap_2D(Q_plus_q_map, energies, S_initial, S_final)

    rho_abs = gaussian_weight(EG_2D, frequencies, sigma_eV, 'absorption')
    rho_emi = gaussian_weight(EG_2D, frequencies, sigma_eV, 'emission')

    energy_cons_term = energy_conservation_term(BED_2D, rho_abs, rho_emi)
    weight_scat_rate_nu_Q_q = tensor_3D_squared * energy_cons_term

    # Integrate out Q and nu to isolate momentum
    rate_mom = np.sum(weight_scat_rate_nu_Q_q, axis=(0, 1))

    # Normalization and FGR Prefactor
    rate_mom_fs = (rate_mom / N_Q) * GOLDEN_RULE_PREFACTOR

    # Extracting directionality of the qpts
    q_x = qpts[:, 0]
    q_y = qpts[:, 1]

    return q_x, q_y, rate_mom_fs





def compute_exciton_momentum_resolved_times(
        g_tensor_5D, energies, frequencies, Q_plus_q_map,
        S_initial, S_final, temp_K, sigma_eV, N_q):
    """
    Computes scattering times (inverse rates) resolved per exciton momentum Q.
    """
    tensor_3D = g_tensor_5D[S_initial, S_final, :, :, :]
    tensor_3D_squared = np.abs(tensor_3D) ** 2

    BED_2D = bose_einstein_dist_2D(temp_K, frequencies)
    EG_2D = energy_gap_2D(Q_plus_q_map, energies, S_initial, S_final)

    rho_abs = gaussian_weight(EG_2D, frequencies, sigma_eV, 'absorption')
    rho_emi = gaussian_weight(EG_2D, frequencies, sigma_eV, 'emission')

    energy_cons_term = energy_conservation_term(BED_2D, rho_abs, rho_emi)
    weight_scat_rate_nu_Q_q = tensor_3D_squared * energy_cons_term

    # Integrate out nu (axis 1) and q (axis 2) to isolate Q (axis 0)
    rate_Q = np.sum(weight_scat_rate_nu_Q_q, axis=(1, 2))

    # Normalize by N_q and apply FGR prefactor
    rate_Q_fs = (rate_Q / N_q) * GOLDEN_RULE_PREFACTOR

    # Safely invert to get time (fs), assigning NaN to forbidden transitions
    safe_rate = np.where(rate_Q_fs < 1e-12, np.nan, rate_Q_fs)
    time_Q_fs = 1.0 / safe_rate

    return time_Q_fs





# --- MAIN EXECUTION BLOCK ---
if __name__ == "__main__":
    print(f"Loading lattice parameters from {raw_energies_file}...")
    recip_lat_data = np.load(raw_energies_file)
    recip_lat_bohr = recip_lat_data['recip_lat']

    #Convert to nm-1 for consistency
    recip_lat_nm = recip_lat_bohr / BOHR_TO_NM

    print(f"Loading data from {xctph_h5}...")
    with h5py.File(xctph_h5, mode='r') as f:
        g_tensor = f['xctph'][:] * RY_TO_EV
        energies = f['energies'][:] * RY_TO_EV
        frequencies = f['frequencies'][:] * RY_TO_EV
        Q_plus_q_map = f['Q_plus_q_map'][:]
        N_Q = f['nQ'][()]
        N_q = f['nq'][()]
        qpts = f['qpts'][:]
        Qpts = f['Qpts'][:]

    print("Applying tensor transformation (frac -> cart)...")
    qpts_cart = np.dot(qpts, recip_lat_nm.T)
    Qpts_cart = np.dot(Qpts, recip_lat_nm.T)

    print("Computing Intraband Rates (S_B -> S_B)...")
    # FIG 2(a)
    energy_x_BB, rate_y_BB = compute_mode_resolved_rates(
        g_tensor, energies, frequencies, Q_plus_q_map,
        bright_exciton_state, bright_exciton_state, temp_K, sigma_eV, N_Q, N_q
    )
    # FIG 2(b)
    q_x_BB, q_y_BB, q_mom_rate_BB = compute_momentum_resolved_rates(
        g_tensor, energies, frequencies, Q_plus_q_map,
        bright_exciton_state, bright_exciton_state, temp_K, sigma_eV, N_Q, qpts
    )
    # FIG 2(c)
    time_Q_BB = compute_exciton_momentum_resolved_times(
        g_tensor, energies, frequencies, Q_plus_q_map,
        bright_exciton_state, bright_exciton_state, temp_K, sigma_eV, N_q
    )

    print("Computing Interband Rates (S_B -> S_D)...")
    # FIG 2(a)
    energy_x_BD, rate_y_BD = compute_mode_resolved_rates(
        g_tensor, energies, frequencies, Q_plus_q_map,
        bright_exciton_state, dark_exciton_state, temp_K, sigma_eV, N_Q, N_q
    )
    # FIG 2(b)
    q_x_BD, q_y_BD, q_mom_rate_BD = compute_momentum_resolved_rates(
        g_tensor, energies, frequencies, Q_plus_q_map,
        bright_exciton_state, dark_exciton_state, temp_K, sigma_eV, N_Q, qpts
    )
    # FIG 2(c)
    time_Q_BD = compute_exciton_momentum_resolved_times(
        g_tensor, energies, frequencies, Q_plus_q_map,
        bright_exciton_state, dark_exciton_state, temp_K, sigma_eV, N_q
    )



    # --- PLOTTING FIG 2(a) ---
    print("Generating rates per phonon mode plots...")
    fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

    # Intraband (Blue)
    ax1.bar(energy_x_BB, rate_y_BB, width=0.002, color='#45B3C4', align='center')
    ax1.set_title(f"Intraband (${S1} \\rightarrow {S1}$)")
    ax1.set_xlabel("$\\hbar\\omega_\\nu$ [eV]")
    ax1.set_ylabel("$\\Gamma_\\nu$ [fs$^{-1}$]")
    ax1.set_xlim(0, 0.10)
    ax1.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

    # Interband (Black)
    ax2.bar(energy_x_BD, rate_y_BD, width=0.002, color='black', align='center')
    ax2.set_title(f"Interband (${S1} \\rightarrow {S2}$)")
    ax2.set_xlabel("$\\hbar\\omega_\\nu$ [eV]")
    ax2.set_xlim(0, 0.10)
    ax2.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

    plt.tight_layout()
    plt.savefig(f"Fig2a_Mode_Resolved_Rates_{S1}_{S2}.png", dpi=300)
    plt.show()
    print(f"Analysis complete. Saved to Fig2a_Mode_Resolved_Rates_{S1}_{S2}.png")




    # --- PLOTTING FIG 2(b) ---
    print("Generating rates per phonon momentum plots...")

    # Create a 1x2 subplot grid
    fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=(12, 6))

    # Set a uniform visual length for all arrows
    arrow_length = 0.5

    # Identify the q_z = 0 plane to prevent 3D arrow overlapping
    unique_qz = np.unique(qpts[:, 2])
    target_qz = unique_qz[np.argmin(np.abs(unique_qz))]
    qz_mask = np.isclose(qpts[:, 2], target_qz)

    # Create a sliced version of qpts for the coordinate search
    qpts_slice = qpts[qz_mask]

    # --- Panel 1: Intraband (SB -> SB) ---
    # Apply the mask to isolate only the target plane
    q_x_BB_slice = q_x_BB[qz_mask]
    q_y_BB_slice = q_y_BB[qz_mask]
    q_mom_rate_BB_slice = q_mom_rate_BB[qz_mask]

    origin_x_BB = np.zeros_like(q_x_BB_slice)
    origin_y_BB = np.zeros_like(q_y_BB_slice)

    # Normalize vectors to a magnitude of 1, then scale to arrow_length
    r_BB = np.sqrt(q_x_BB_slice ** 2 + q_y_BB_slice ** 2)
    r_BB_safe = np.where(r_BB == 0, 1, r_BB)
    q_x_BB_norm = (q_x_BB_slice / r_BB_safe) * arrow_length
    q_y_BB_norm = (q_y_BB_slice / r_BB_safe) * arrow_length

    # Extract bounds from the sliced data
    vmin_BB = np.min(q_mom_rate_BB_slice)
    vmax_BB = np.max(q_mom_rate_BB_slice)

    quiver_BB = ax3.quiver(origin_x_BB, origin_y_BB, q_x_BB_norm, q_y_BB_norm, q_mom_rate_BB_slice,
                           cmap='Blues', angles='xy', scale_units='xy', scale=1,
                           norm=plt.Normalize(vmin=vmin_BB, vmax=vmax_BB))

    # Add a dot at q=0 to represent the origin rate visually
    gamma_idx_BB = np.where((qpts_slice[:, 0] == 0) & (qpts_slice[:, 1] == 0) & (qpts_slice[:, 2] == 0))[0][0]
    rate_gamma_BB = q_mom_rate_BB_slice[gamma_idx_BB]
    ax3.scatter(0, 0, c=[rate_gamma_BB], cmap='Blues', norm=plt.Normalize(vmin=vmin_BB, vmax=vmax_BB),
                s=60, zorder=3, edgecolors='black', linewidth=0.5)

    # Text annotations for the rates and axes
    ax3.text(0.02, 0.02, rf"$\Gamma_{{q=0}} = {rate_gamma_BB:.2e} \ fs^{{-1}}$",
             transform=ax3.transAxes, fontsize=10, verticalalignment='bottom')
    ax3.text(0.55, 0.0, '$q_x$', fontsize=12, va='center', ha='left')
    ax3.text(0.0, 0.55, '$q_y$', fontsize=12, ha='center', va='bottom')

    ax3.set_title(f"Intraband (${S1} \\rightarrow {S1}$) | $q_z = {target_qz:.2f}$")
    ax3.set_xlim(-0.6, 0.6)
    ax3.set_ylim(-0.6, 0.6)
    ax3.set_aspect('equal')
    ax3.axis('off')

    cbar_BB = fig2.colorbar(quiver_BB, ax=ax3, orientation='horizontal', shrink=0.6, pad=0.05)
    formatter_BB = ticker.ScalarFormatter(useMathText=False)
    formatter_BB.set_powerlimits((0, 0))
    cbar_BB.ax.xaxis.set_major_formatter(formatter_BB)
    cbar_BB.set_label(rf'$\Gamma_q^{{{S1} {S1}}} \ [fs^{{-1}}]$')

    # --- Panel 2: Interband (SB -> SD) ---
    # Apply the mask to isolate only the target plane
    q_x_BD_slice = q_x_BD[qz_mask]
    q_y_BD_slice = q_y_BD[qz_mask]
    q_mom_rate_BD_slice = q_mom_rate_BD[qz_mask]

    origin_x_BD = np.zeros_like(q_x_BD_slice)
    origin_y_BD = np.zeros_like(q_y_BD_slice)

    # Normalize vectors to a magnitude of 1, then scale to arrow_length
    r_BD = np.sqrt(q_x_BD_slice ** 2 + q_y_BD_slice ** 2)
    r_BD_safe = np.where(r_BD == 0, 1, r_BD)
    q_x_BD_norm = (q_x_BD_slice / r_BD_safe) * arrow_length
    q_y_BD_norm = (q_y_BD_slice / r_BD_safe) * arrow_length

    # Extract bounds from the sliced data
    vmin_BD = np.min(q_mom_rate_BD_slice)
    vmax_BD = np.max(q_mom_rate_BD_slice)

    quiver_BD = ax4.quiver(origin_x_BD, origin_y_BD, q_x_BD_norm, q_y_BD_norm, q_mom_rate_BD_slice,
                           cmap='Greys', angles='xy', scale_units='xy', scale=1,
                           norm=plt.Normalize(vmin=vmin_BD, vmax=vmax_BD))

    # Add a dot at q=0 to represent the origin rate visually
    gamma_idx_BD = np.where((qpts_slice[:, 0] == 0) & (qpts_slice[:, 1] == 0) & (qpts_slice[:, 2] == 0))[0][0]
    rate_gamma_BD = q_mom_rate_BD_slice[gamma_idx_BD]
    ax4.scatter(0, 0, c=[rate_gamma_BD], cmap='Greys', norm=plt.Normalize(vmin=vmin_BD, vmax=vmax_BD),
                s=60, zorder=3, edgecolors='black', linewidth=0.5)

    # Text annotations for the rates and axes
    ax4.text(0.02, 0.02, rf"$\Gamma_{{q=0}} = {rate_gamma_BD:.2e} \ fs^{{-1}}$",
             transform=ax4.transAxes, fontsize=10, verticalalignment='bottom')
    ax4.text(0.55, 0.0, '$q_x$', fontsize=12, va='center', ha='left')
    ax4.text(0.0, 0.55, '$q_y$', fontsize=12, ha='center', va='bottom')

    ax4.set_title(f"Interband (${S1} \\rightarrow {S2}$) | $q_z = {target_qz:.2f}$")
    ax4.set_xlim(-0.6, 0.6)
    ax4.set_ylim(-0.6, 0.6)
    ax4.set_aspect('equal')
    ax4.axis('off')

    cbar_BD = fig2.colorbar(quiver_BD, ax=ax4, orientation='horizontal', shrink=0.6, pad=0.05)
    formatter_BD = ticker.ScalarFormatter(useMathText=False)
    formatter_BD.set_powerlimits((0, 0))
    cbar_BD.ax.xaxis.set_major_formatter(formatter_BD)
    cbar_BD.set_label(rf'$\Gamma_q^{{{S1} {S2}}} \ [fs^{{-1}}]$')

    plt.tight_layout()
    plt.savefig(f"DEBUGGING2_Fig2b_Momentum_Resolved_Rates_{S1}_{S2}.png", dpi=300)
    plt.show()
    print(f"Analysis complete. Saved to Fig2b_Momentum_Resolved_Rates_{S1}_{S2}.png")






    # --- PLOTTING FIG 2(c) ---
    print("Generating scattering times per exciton momentum plots...")

    # Extract unique coordinates from the EXCITON momentum grid
    unique_Qx = np.unique(Qpts[:, 0])
    unique_Qy = np.unique(Qpts[:, 1])
    unique_Qz = np.unique(Qpts[:, 2])

    # Find the Qz plane closest to 0
    target_Qz = unique_Qz[np.argmin(np.abs(unique_Qz))]

    # Calculate grid extents for imshow
    dQX = (unique_Qx[1] - unique_Qx[0]) if len(unique_Qx) > 1 else 1.0
    dQY = (unique_Qy[1] - unique_Qy[0]) if len(unique_Qy) > 1 else 1.0
    extent = [unique_Qx.min() - dQX / 2, unique_Qx.max() + dQX / 2,
              unique_Qy.min() - dQY / 2, unique_Qy.max() + dQY / 2]

    # Map the 1D arrays onto 2D matrices for the chosen Qz slice
    time_2D_BB = np.zeros((len(unique_Qy), len(unique_Qx)))
    time_2D_BD = np.zeros((len(unique_Qy), len(unique_Qx)))

    for idx in range(N_Q):
        # Only map points that belong to the target Qz plane
        if np.isclose(Qpts[idx, 2], target_Qz):
            Qx_val = Qpts[idx, 0]
            Qy_val = Qpts[idx, 1]

            # Find the matrix indices
            ix = np.where(unique_Qx == Qx_val)[0][0]
            iy = np.where(unique_Qy == Qy_val)[0][0]

            time_2D_BB[iy, ix] = time_Q_BB[idx]
            time_2D_BD[iy, ix] = time_Q_BD[idx]


    # Create the plot
    fig3, (ax5, ax6) = plt.subplots(1, 2, figsize=(10, 4))

    # Tell the colormaps to color NaN (forbidden) values clearly
    cmap_BB = plt.cm.GnBu_r.copy()
    cmap_BB.set_bad(color='crimson')  # Highlights forbidden states in red

    # Changed from 'gray_r' to 'gray' so that faster times (low fs) are dark
    cmap_BD = plt.cm.gray.copy()
    cmap_BD.set_bad(color='crimson')

    # Intraband
    im_BB = ax5.imshow(time_2D_BB, origin='lower', cmap=cmap_BB, extent=extent)
    ax5.set_title(f"Intraband (${S1} \\rightarrow {S1}$) | $Q_z = {target_Qz:.2f}$")
    ax5.set_xlabel("Qx [crys]")
    ax5.set_ylabel("Qy [crys]")
    cbar_BB = fig3.colorbar(im_BB, ax=ax5)
    cbar_BB.set_label(rf'$\Gamma_Q^{{{S1} {S1} -1}}$ [fs]')

    # Interband
    im_BD = ax6.imshow(time_2D_BD, origin='lower', cmap=cmap_BD, extent=extent)
    ax6.set_title(f"Interband (${S1} \\rightarrow {S2}$) | $Q_z = {target_Qz:.2f}$")
    ax6.set_xlabel("Qx [crys]")
    cbar_BD = fig3.colorbar(im_BD, ax=ax6)
    cbar_BD.set_label(rf'$\Gamma_Q^{{{S1} {S2} -1}}$ [fs]')

    plt.tight_layout()
    plt.savefig(f"Fig2c_Exciton_Momentum_Rates_{S1}_{S2}.png", dpi=300)
    plt.show()
    print(f"Analysis complete. Saved to Fig2c_Exciton_Momentum_Rates_{S1}_{S2}.png")

