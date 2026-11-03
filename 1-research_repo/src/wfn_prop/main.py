import numpy as np
import argparse
from wfn_prop.NumMthds import MultiBandRungeKutta4, CentralDifference3d
from wfn_prop.k_scat import MultiBandTransitionMatrix
import sys
import os
from wfn_prop.io import parse_input_file

# --- DUAL STREAM LOGGER FOR CLUSTER INTERCEPTION ---
class DualStream:
    """
    Intersects and mirrors all standard console print outputs into an external log file.
    Ensures prints inside imported modules are cleanly preserved in output.log.
    """
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.log = open(filepath, 'w', encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()


# --- STYLIZED SIMULATION BANNER ---
def print_banner():
    """Prints a structured ASCII execution banner acknowledging development and core physics literature."""
    ascii_art = r"""
 __        __  __          ____                 
 \ \      / / / _|        |  _ \ _ __ ___  _ __ 
  \ \ /\ / / | |_  _ __   | |_) | '__/ _ \| '_ \
   \ V  V /  |  _|| '_ \  |  __/| | | (_) | |_) |
    \_/\_/   |_|  |_| |_| |_|   |_|  \___/| .__/ 
                                          |_|    
    """
    w = 62
    print(ascii_art)
    print("*" * w)
    print("*" + "Boltzmann Transport + K_Scattering".center(w - 2) + "*")
    print("*" + "v2.0 - N-Band Modular Architecture".center(w - 2) + "*")
    print("*" * w)
    print("*" + "".center(w - 2) + "*")
    print("*" + " Exciton-Phonon Dynamics Framework Derived From:".ljust(w - 2) + "*")
    print("*" + "   Cohen, Haber, Neaton, Qiu, & Refaely-Abramson".ljust(w - 2) + "*")
    print("*" + "   Phys. Rev. Lett. 132, 126902 (2024)".ljust(w - 2) + "*")
    print("*" + "".center(w - 2) + "*")
    print("*" + " Code Written & Optimized by:".ljust(w - 2) + "*")
    print("*" + "   Samson Baughman".ljust(w - 2) + "*")
    print("*" + "".center(w - 2) + "*")
    print("*" * w)
    print("\n")

# Gaussian Distribution Function
def gaussian_dist_2d(x_pos, y_pos, spread_x, spread_y, amplitude, center_x, center_y):
    term1 = - ( (x_pos - center_x)**2 ) / (2 * spread_x**2)
    term2 = - ( (y_pos - center_y)**2 ) / (2 * spread_y**2)
    dist_funct = amplitude * np.exp( term1 + term2 )
    return dist_funct

# Initializing Gaussian Wavepack
# From: Signatures of Dimensionality and Symmetry in Exciton Band
#       Structure: Consequences for Exciton Dynamics and Transport
def init_gaussian_spread_2D(X_grid, Y_grid, Q_vectors, grid_x, grid_y, x_0, y_0, sigma_R, sigma_Q, amplitude = 1.0):
    """Initializes bright exciton phase-space wavepacket with a momentum-space spread."""
    print(f"Initializing dynamic Gaussian spread (sigma_R={sigma_R} nm, sigma_Q={sigma_Q} nm^-1)...")
    N_Q = len(Q_vectors)
    occupation_matrix_3d = np.zeros((grid_x, grid_y, N_Q), dtype=np.float32)

    for q_index in range(N_Q):
        Q_vector = Q_vectors[q_index]
        magnitude_Q_sq = np.sum(Q_vector ** 2)
        weight_Q = np.exp(-magnitude_Q_sq / (2 * sigma_Q ** 2))

        spatial_dist = gaussian_dist_2d(X_grid, Y_grid, spread_x=sigma_R, spread_y=sigma_R,
                                        amplitude=amplitude, center_x=x_0, center_y=y_0)
        occupation_matrix_3d[:, :, q_index] = spatial_dist * weight_Q

    return occupation_matrix_3d


def init_single_q_2D(X_grid, Y_grid, N_Q, grid_x, grid_y, x_0, y_0, sigma_R, target_q_index, amplitude=1.0):
    """Initializes 100% of the wavepacket mass in a single chosen Q-index."""
    # STRICT BOUNDS CHECKING
    if target_q_index < 0 or target_q_index >= N_Q:
        print(f"\nFATAL ERROR: Target Q-index {target_q_index} is out of bounds!")
        sys.exit(1)

    print(f"Initializing wavepacket strictly at single Q-index {target_q_index}...")
    occupation_matrix_3d = np.zeros((grid_x, grid_y, N_Q), dtype=np.float32)

    spatial_dist = gaussian_dist_2d(X_grid, Y_grid, spread_x=sigma_R, spread_y=sigma_R,
                                    amplitude=amplitude, center_x=x_0, center_y=y_0)

    # Drop all the mass into the targeted index slice
    occupation_matrix_3d[:, :, target_q_index] = spatial_dist * 1.0
    return occupation_matrix_3d

# --- Main Function ---
def main():
    parser = argparse.ArgumentParser(description="Run Exciton Transport via Input File")
    parser.add_argument("config_file", type=str, help="Path to the simulation config.txt")
    args = parser.parse_args()

    # --- INITIALIZE DUAL LOGGING STREAM ---
    logger = DualStream("output.log")
    sys.stdout = logger

    # Print the Wfn Prop header banner
    print_banner()

    # Parse parameters dynamically from text file
    print(f"Reading configuration from {args.config_file}...")
    cfg = parse_input_file(args.config_file)

    # --- PARAMETER METADATA LOG DUMP ---
    w = 62
    print("=" * w)
    print("RUN CONFIGURATION PARAMETERS LOGGED:")
    print("=" * w)
    for key, val in sorted(cfg.items()):
        print(f"  {key.ljust(20)} = {val}")
    print("=" * w + "\n")

    # Spatial Setup
    delta_x = cfg['length_x'] / (cfg['grid_x'] - 1)
    delta_y = cfg['length_y'] / (cfg['grid_y'] - 1)
    x_0, y_0 = cfg['length_x'] / 2.0, cfg['length_y'] / 2.0

    # Grid Allocation (float32 for memory optimization)
    X_grid = np.empty((cfg['grid_x'], cfg['grid_y']), dtype=np.float32)
    Y_grid = np.empty((cfg['grid_x'], cfg['grid_y']), dtype=np.float32)
    for i in range(cfg['grid_x']):
        for j in range(cfg['grid_y']):
            X_grid[i, j] = i * delta_x
            Y_grid[i, j] = j * delta_y

    # Load pre-compiled physics files straight from the config dictionary
    print(f"Loading physics payload: {cfg['physics_file']}")
    physics_payload = np.load(cfg['physics_file'])

    print(f"Loading velocity payload: {cfg['vel_file']}")
    velocity_payload = np.load(cfg['vel_file'])

    Q_vectors = physics_payload['Qpts']
    N_Q = len(Q_vectors)

    # Failsafe: check mapping
    if not np.allclose(Q_vectors, velocity_payload['Qpts']):
        raise ValueError("FATAL: Momentum ordering mismatch between your configuration's input payloads!")

    # Wavepacket Initialization (Base 3D Slice)
    init_mode = cfg.get('init_mode', 'gauss')
    if init_mode == 'gauss':
        occupation_matrix_3d = init_gaussian_spread_2D(
            X_grid, Y_grid, Q_vectors, cfg['grid_x'], cfg['grid_y'],
            x_0, y_0, cfg['sigma_R'], cfg['sigma_Q'], cfg['amplitude']
        )
    elif init_mode == 'single-q':
        occupation_matrix_3d = init_single_q_2D(
            X_grid, Y_grid, N_Q, cfg['grid_x'], cfg['grid_y'],
            x_0, y_0, cfg['sigma_R'], cfg.get('target_q_index', 0), cfg['amplitude']
        )

    # --- 1. PARSE N-BAND CONFIGURATION ---
    state_names = cfg['state_names']
    advection_flags = cfg['state_advection']
    gamma_decays = cfg['state_gamma_decay']

    if isinstance(state_names, str): state_names = (state_names,)
    if isinstance(advection_flags, (bool, str)): advection_flags = (advection_flags,)
    if isinstance(gamma_decays, (float, int)): gamma_decays = (gamma_decays,)

    n_bands = len(state_names)
    print(f"\nAssembling {n_bands}-Band Phase Space Tensor for states: {state_names}")

    # --- 2. BUILD 4D TENSOR ---
    n_initial_4d = np.zeros((n_bands, cfg['grid_x'], cfg['grid_y'], N_Q), dtype=np.float32)
    n_initial_4d[0] = occupation_matrix_3d

    # --- 3. DYNAMICALLY ASSEMBLE SPATIAL SOLVERS ---
    spatial_solvers = []
    for i, state in enumerate(state_names):
        flag = str(advection_flags[i]).strip().lower() == 'true'
        if flag:
            print(f"  [{state}] Advection ENABLED. Loading velocities.")
            vx = velocity_payload[f'vel_x_{state}'].astype(np.float32)
            vy = velocity_payload[f'vel_y_{state}'].astype(np.float32)
        else:
            print(f"  [{state}] Advection DISABLED. Pinning spatial grid.")
            vx = np.zeros(N_Q, dtype=np.float32)
            vy = np.zeros(N_Q, dtype=np.float32)

        spatial_solvers.append(CentralDifference3d(dx=delta_x, dy=delta_y, vel_x=vx, vel_y=vy))

    # --- 4. DYNAMICALLY COLLECT SCATTERING RATES ---
    rates_dict = {}
    for out_idx, out_state in enumerate(state_names):
        for in_idx, in_state in enumerate(state_names):
            key = f'Rate_{out_state}_{in_state}'
            if key in physics_payload:
                rates_dict[(out_idx, in_idx)] = physics_payload[key]
            elif out_idx != in_idx:
                print(f"  Warning: No interband scattering matrix found for {key}. Assuming 0.0.")

    print("\nInitializing Multi-Band Scattering Operator...")
    scattering_obj = MultiBandTransitionMatrix(
        n_bands=n_bands,
        rates_dict=rates_dict,
        gamma_decay_rates=gamma_decays,
        map_Q_to_q=physics_payload['Q_plus_q_map'],
        gamma_index=physics_payload['gamma_index']
    )

    # --- 5. TIME INTEGRATION ---
    print("\nInitializing Runge-Kutta 4 Integrator...")
    time_integrator = MultiBandRungeKutta4(
        spatial_solvers=spatial_solvers,
        total_sim_time=cfg['sim_time'],
        scattering_solver=scattering_obj
    )

    # Save standard frames array
    frames = time_integrator.solve(n_initial_4d, save_interval=2)
    output_data_path = f"./{cfg['gif_filename']}_frames.npz"
    np.savez_compressed(output_data_path, frames=np.array(frames), dt=time_integrator.dt)
    print(f"SUCCESS: Raw frames exported to {output_data_path}.")

    # Stream to HDF5
    hdf5_data_path = time_integrator.solve_hdf5_compressed(
        n_initial_4d,
        save_interval=2,
        output_file=f"./{cfg['gif_filename']}_frames.h5"
    )
    print(f"SUCCESS: HDF5 stream exported to {hdf5_data_path}.")

    print("\n" + "=" * w)
    print("Job Complete".center(w))
    print("=" * w)

    # Close file streams cleanly
    sys.stdout = logger.terminal
    logger.close()


if __name__ == "__main__":
    main()