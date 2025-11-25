import math
import numpy as np
from ase import Atoms
from ase.io.vasp import write_vasp
from ase.io import write as write_generic
import pandas as pd
import os
import glob
import sys


# ------------------------------------------------------------------
# ZG FUNCTIONS
# ------------------------------------------------------------------

def bose_einstein_occupation(freq_hz, temperature_k):
    """Calculates Bose-Einstein occupation for a given frequency (in Hz) and temperature."""
    if temperature_k == 0:
        return 0.0
    boltzmann_constant = 1.38065e-23  # Joules per Kelvin
    reduced_planck = 1.05457e-34  # Joules times second
    ang_freq = freq_hz * 2 * math.pi

    try:
        exp_arg = (reduced_planck * ang_freq) / (boltzmann_constant * temperature_k)
        if exp_arg > 700:  # np.exp(700) is ~1e304
            return 0.0  # Occupation is essentially zero

        exp1 = np.exp(exp_arg)
        if exp1 == 1.0:
            if ang_freq == 0: return 0.0  # Handle 0 freq
            n = (boltzmann_constant * temperature_k) / (reduced_planck * ang_freq)
        else:
            n = (exp1 - 1) ** -1
        return n
    except OverflowError:
        print(f"Warning: Overflow encountered in Bose-Einstein for freq {freq_hz} Hz.")
        return 0.0


def tau_displacement(atomic_mass_kg, freq_hz, temperature_k):
    """Calculates the thermal displacement (tau) for a given mass, freq, and temp."""
    if freq_hz == 0.0:
        return 0.0

    reduced_planck = 1.05457e-34
    bose_ein = bose_einstein_occupation(freq_hz, temperature_k)
    ang_freq = freq_hz * 2 * math.pi

    m1 = 2 * bose_ein + 1
    m2 = reduced_planck / (2 * atomic_mass_kg * ang_freq)
    disp = (m1 * m2) ** (1 / 2) #in meters
    disp_ang = disp * 1e10 #conversion to angstroms

    return disp_ang


def write_qe_snippet(filename, atoms, coordinate_type='cartesian'):
    """
    Manually writes a QE snippet for cell and positions as a fallback.
    This is more robust than the 'espresso-in' writer which can fail.
    """
    print(f"    -> ASE writer failed. Using manual fallback for: {os.path.basename(filename)}")
    with open(filename, 'w') as f:
        # 1. Write Cell
        f.write("CELL_PARAMETERS (angstrom)\n")
        cell = atoms.get_cell()
        for i in range(3):
            f.write(f"  {cell[i][0]:16.16f}  {cell[i][1]:16.16f}  {cell[i][2]:16.16f}\n")

        f.write("\n")

        # 2. Write Atoms
        symbols = atoms.get_chemical_symbols()
        nat = atoms.get_global_number_of_atoms()

        if coordinate_type.lower() == 'cartesian':
            f.write(f"ATOMIC_POSITIONS (angstrom)\n")
            positions = atoms.get_positions()
        else:  # fractional
            f.write(f"ATOMIC_POSITIONS (crystal)\n")
            positions = atoms.get_scaled_positions()

        for i in range(nat):
            f.write(f"  {symbols[i]:<3}  {positions[i][0]:16.16f}  {positions[i][1]:16.16f}  {positions[i][2]:16.16f}\n")


def process_phonon_file(
        input_csv_file: str,
        atom_mass_map: list,
        nat: int,
        atom_string: str,
        phonon_freq_hz: float,
        temperature_k: float,
        displacement_percentage: float,
        cell_matrix: list
):
    """
    Reads a single phonon CSV, calculates displacements, and returns an
    ASE Atoms object and a detailed DataFrame.
    """
    try:
        df = pd.read_csv(input_csv_file)
    except FileNotFoundError:
        print(f"  Error: Input CSV file not found at '{input_csv_file}'")
        return None, None

    df_list = df.values.tolist()

    if len(df_list) != nat:
        print(f"  Error: CSV file has {len(df_list)} rows, but atom counts specify {nat} atoms.")
        return None, None

    col_names = ['X_T_dep', 'Y_T_dep', 'Z_T_dep', 'Eigenvectors_(dx)_T_dep', 'Eigenvectors_(dy)_T_dep',
                 'Eigenvectors_(dz)_T_dep', 'X', 'Y', 'Z', 'Eigenvectors_(dx)', 'Eigenvectors_(dy)',
                 'Eigenvectors_(dz)']
    pos_data = []

    for k in range(nat):
        unmod_pos = df_list[k][:3]
        unmod_eig = df_list[k][3:]
        current_mass_kg = atom_mass_map[k]
        mod_eig_list = []
        temp_dep_pos_list = []

        for j in range(3):
            mod_eig = tau_displacement(current_mass_kg, phonon_freq_hz, temperature_k) * unmod_eig[j]
            pos_temp_dep = unmod_pos[j] + (mod_eig * displacement_percentage)
            mod_eig_list.append(mod_eig)
            temp_dep_pos_list.append(pos_temp_dep)

        pos_data_list = temp_dep_pos_list + mod_eig_list + unmod_pos + unmod_eig
        pos_data.append(pos_data_list)

    detailed_df = pd.DataFrame(pos_data, columns=col_names)

    new_pos_array = np.array([row[0:3] for row in pos_data])
    atoms = Atoms(atom_string)
    atoms.set_positions(new_pos_array)
    atoms.set_cell(cell_matrix)
    atoms.pbc = True

    return atoms, detailed_df


# ------------------------------------------------------------------
# SCRIPT EXECUTION (MAIN)
# ------------------------------------------------------------------
def main():
    """
    Main function to run the batch processing.
    Edit all parameters in the 'USER SETTINGS' section below.
    """

    # ------------------------------------------------------------------
    # USER SETTINGS (!!! EDIT THIS SECTION !!!)
    # ------------------------------------------------------------------

    SYSTEM_NAME = "deut-pent"
    ATOM_SYMBOLS = ['C', 'H']
    ATOM_COUNTS = [44, 28]
    ATOM_MASSES_KG = [
        1.9926468e-26,  # Mass of Carbon in kg
        1.6735575e-27  # Mass of Hydrogen in kg
    ]
    TEMPERATURE_K = 300
    DISPLACEMENT_PERC = 1 #Ex: 0.3 ~ 30%

    # Point this to your CSV files.
    # For example, if they are in a subfolder 'my_csvs':
    # INPUT_FILE_PATTERN = "my_csvs/*_cm-1_posvec+eigvec.csv"
    # If they are in the *same* folder as the script:
    # INPUT_FILE_PATTERN = "*_cm-1_posvec+eigvec.csv"
    INPUT_FILE_PATTERN = f"../VASP_FREQ_OUTCAR_to_csv/freq_csv_{SYSTEM_NAME}/*_cm-1_posvec+eigvec.csv"

    MASTER_OUTPUT_FOLDER = f"master_freq_folder_{SYSTEM_NAME}"

    CELL_VECTORS = [
        [6.2660000000000000, 0.0000000000000000, 0.0000000000000000],
        [0.7203431964649767, 7.7415586724707204, 0.0000000000000000],
        [0.5876759734010517, 3.3581219057453895, 14.1243957115495764]
    ]

    # ------------------------------------------------------------------
    # END OF USER SETTINGS
    # ------------------------------------------------------------------

    print("--- Starting Batch Phonon Processor (v2) ---")

    if not (len(ATOM_SYMBOLS) == len(ATOM_COUNTS) == len(ATOM_MASSES_KG)):
        print("Error: 'atom_symbols', 'atom_counts', and 'atom_masses_kg' lists must have the same length.")
        sys.exit(1)

    mass_map = []
    atom_string_list = []
    for symbol, count, mass in zip(ATOM_SYMBOLS, ATOM_COUNTS, ATOM_MASSES_KG):
        mass_map.extend([mass] * count)
        atom_string_list.extend([symbol] * count)

    nat = sum(ATOM_COUNTS)
    atom_string = "".join(atom_string_list)
    print(f"System: {atom_string} (Total atoms: {nat})")
    print(f"Temperature: {TEMPERATURE_K} K, Displacement: {DISPLACEMENT_PERC * 100}%")

    csv_files = glob.glob(INPUT_FILE_PATTERN)
    if not csv_files:
        print(f"Error: No files found matching pattern '{INPUT_FILE_PATTERN}'")
        print("Please check the INPUT_FILE_PATTERN setting.")
        sys.exit(1)

    print(f"Found {len(csv_files)} files to process.")

    CM_TO_HZ = 29979245800.0
    perc_str = f"{DISPLACEMENT_PERC * 100:.0f}perc"

    for i, file_path in enumerate(csv_files):
        print(f"\n--- Processing File {i + 1}/{len(csv_files)}: {os.path.basename(file_path)} ---")

        file_name = os.path.basename(file_path)
        try:
            freq_cm_str = file_name.split('_cm-1')[0]
            freq_cm = float(freq_cm_str)
            freq_hz = freq_cm * CM_TO_HZ
            print(f"freq_cm: {freq_cm} Hz")
            print(f"freq_hz: {freq_hz} Hz") ######
            freq_folder_name = f"{freq_cm_str}"
            print(f"  Freq: {freq_cm_str} cm-1 ({freq_hz:.3e} Hz)")
        except (IndexError, ValueError):
            print(f"  Warning: Could not parse frequency from filename: {file_name}")
            print("  Skipping this file.")
            continue

        qe_path = os.path.join(MASTER_OUTPUT_FOLDER, freq_folder_name, "QE")
        vasp_path = os.path.join(MASTER_OUTPUT_FOLDER, freq_folder_name, "VASP")
        csv_path = os.path.join(MASTER_OUTPUT_FOLDER, freq_folder_name, "CSV")

        os.makedirs(qe_path, exist_ok=True)
        os.makedirs(vasp_path, exist_ok=True)
        os.makedirs(csv_path, exist_ok=True)

        atoms, detailed_df = process_phonon_file(
            input_csv_file=file_path,
            atom_mass_map=mass_map,
            nat=nat,
            atom_string=atom_string,
            phonon_freq_hz=freq_hz,
            temperature_k=TEMPERATURE_K,
            displacement_percentage=DISPLACEMENT_PERC,
            cell_matrix=CELL_VECTORS
        )

        if atoms is None:
            print(f"  Failed to process file {file_name}. Skipping.")
            continue

        base_filename = f"{freq_folder_name}_mod_{perc_str}"

        # --- VASP (Isolated block) ---
        try:
            vasp_cart_file = os.path.join(vasp_path, base_filename + ".vasp_cart")
            write_vasp(vasp_cart_file, atoms, direct=False, vasp5=True)

            vasp_frac_file = os.path.join(vasp_path, base_filename + ".vasp_frac")
            write_vasp(vasp_frac_file, atoms, direct=True, vasp5=True)
            print("  VASP files... OK")
        except Exception as e:
            print(f"  Error writing VASP files: {e}")

        # --- Quantum ESPRESSO (Isolated block with fallback) ---
        qe_cart_file = os.path.join(qe_path, base_filename + ".qe_cart")
        qe_frac_file = os.path.join(qe_path, base_filename + ".qe_frac")
        try:
            # Try ASE's built-in writer first
            write_generic(qe_cart_file, atoms, format='espresso-in', crystal_coordinates=False)
            write_generic(qe_frac_file, atoms, format='espresso-in', crystal_coordinates=True)
            print("  QE files (ASE)... OK")
        except Exception as e:
            print(f"  Warning: ASE 'espresso-in' writer failed ({e}).")
            # Fallback to manual writer
            try:
                write_qe_snippet(qe_cart_file, atoms, coordinate_type='cartesian')
                write_qe_snippet(qe_frac_file, atoms, coordinate_type='fractional')
                print("  QE files (Manual Fallback)... OK")
            except Exception as e_manual:
                print(f"  Error: Manual QE writer also failed: {e_manual}")

        # --- CSV Files (Isolated block) ---
        try:
            csv_detailed_file = os.path.join(csv_path, base_filename + "_detailed_log.csv")
            detailed_df.to_csv(csv_detailed_file, index=False)

            csv_cart_file = os.path.join(csv_path, base_filename + "_cartesian_coords.csv")
            cart_df = detailed_df[['X_T_dep', 'Y_T_dep', 'Z_T_dep']]
            cart_df.columns = ['X_cart', 'Y_cart', 'Z_cart']
            cart_df.to_csv(csv_cart_file, index=False)

            csv_frac_file = os.path.join(csv_path, base_filename + "_fractional_coords.csv")
            frac_coords = atoms.get_scaled_positions()
            frac_df = pd.DataFrame(frac_coords, columns=['Frac_X', 'Frac_Y', 'Frac_Z'])
            frac_df.to_csv(csv_frac_file, index=False)
            print("  CSV files... OK")
        except Exception as e:
            print(f"  Error writing CSV files: {e}")

        print(f"  Finished processing: {os.path.join(MASTER_OUTPUT_FOLDER, freq_folder_name)}")

    print("\n--- Batch processing complete ---")


if __name__ == "__main__":
    main()