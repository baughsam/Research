import argparse
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Package modules
from ct_character.IOHandler import IOHandler
from ct_character.Solver import Solver
from ct_character.Exciton import ExcitonData, Configuration


def parse_input_file(filepath: Path):
    """
    Parses 7-line master input file format.
    """
    try:
        with open(filepath, 'r') as f:
            lines = [line.strip() for line in f if line.strip()]

        if len(lines) < 6:
            raise ValueError(f"Input file must have at least 6 lines Found {len(lines)}.")

        # Line 1: Cube filename
        cube_file = lines[0]

        # Line 2: shape Type
        shape_type = lines[1]

        # Line 6: Output Prefix
        output_prefix = lines[5]

        # Line 7 (Optional Toggle for INVOLUME file)
        do_rdf = False
        if len(lines) >= 7:
            val = lines[6].lower()
            if val in ['true', 't', 'yes', 'on', '1']:
                do_rdf = True
            print(f"  > In-Volume/RDF Analysis: {'ENABLED' if do_rdf else 'DISABLED'}")

        # Lines 3-5: Shape Dimensions
        shape_params = {}

        # Raw lines for the parameters
        param_lines = lines[2:5]

        try:
            # Check Shape Type to decide how to parse
            if shape_type in ['Parallelepiped', 'Box']:
                print(f"  > Parsing vector parameters for {shape_type}...")
                shape_params = {
                    "vec_a": np.fromstring(param_lines[0], sep=" "),
                    "vec_b": np.fromstring(param_lines[1], sep=" "),
                    "vec_c": np.fromstring(param_lines[2], sep=" "),
                }
                # Validation: Ensure vectors are size 3
                if any(v.size != 3 for v in shape_params.values()):
                    print("Warning: Vectors must have 3 components. Using Defaults.")
                    shape_params = {}

            else:
                # Default behavior: scalar floats (Cylinders, Spheres, etc...)
                print(f"  > Parsing scalar parameters for {shape_type}...")
                shape_params = {
                    "axis_a": float(param_lines[0]),
                    "axis_b": float(param_lines[1]),
                    "length_c": float(param_lines[2]),
                }
        except ValueError as e:
            print(f"Warning: Could not parse parameters for '{shape_type}': {e}")
            print("  > Will attempt to use Shape defaults.")
            shape_params = {}

        return cube_file, shape_type, shape_params, output_prefix, do_rdf

    except Exception as e:
        print(f"Error parsing input file '{filepath}'. {e}")
        sys.exit(1)


def print_banner():
    ascii_art = r"""
      ______ ______   ___                __            _
     / ____//_  __/  /   |  ____  ____ _/ /_  _______ (_)____
    / /      / /    / /| | / __ \/ __ `/ / / / / ___// // ___/
   / /___   / /    / ___ |/ / / / /_/ / / /_/ (__  )/ (__  )
   \____/  /_/    /_/  |_/_/ /_/\__,_/_/\__, /____//_/____/
                                       /____/
    """

    w = 62

    print(ascii_art)
    print("*" * w)
    print("*" + "EXCITON CHARACTERIZATION SUITE".center(w - 2) + "*")
    print("*" + "v1.0.0 (Python Port)".center(w - 2) + "*")
    print("*" * w)
    print("*" + " ".center(w - 2) + "*")
    print("*" + "  Based on the original Fortran implementation by:".ljust(w - 2) + "*")
    print("*" + "    Sahar Sharifzadeh & Pierre Darancet".ljust(w - 2) + "*")
    print("*" + "    (The Molecular Foundry, Berkeley)".ljust(w - 2) + "*")
    print("*" + " ".center(w - 2) + "*")
    print("*" + "  Python Rewrite & Optimization by:".ljust(w - 2) + "*")
    print("*" + "    Samson Baughman".ljust(w - 2) + "*")
    print("*" + " ".center(w - 2) + "*")
    print("*" * w)
    print("*" + "  This software assumes the GNU General Public License.".ljust(w - 2) + "*")
    print("*" + "  See http://www.gnu.org/copyleft/gpl.txt".ljust(w - 2) + "*")
    print("*" * w)
    print("\n")


def main():
    print_banner()

    parser = argparse.ArgumentParser(description="Charge Transfer Analysis Code")
    parser.add_argument("input_file", type=str, help="Path to the master input file (e.g., INPUT_CTCALC.in)")

    # NEW ARGUMENT FLAG
    parser.add_argument("--print-analysis-graph", action='store_true',
                        help="Output the normalized volume-corrected probability density plot")
    args = parser.parse_args()

    input_path = Path(args.input_file)
    if not input_path.exists():
        print(f"Error: Input file '{input_path}' not found.")
        sys.exit(1)

    print(f"--- Starting CT analysis on {input_path} ---")

    cube_filename, shape_type, shape_params, out_prefix, do_rdf = parse_input_file(input_path)

    print(f"  Target Cube File: {cube_filename}")
    print(f"  Shape Model:      {shape_type}")

    cube_path = Path(cube_filename)
    if not cube_path.is_absolute():
        cube_path = input_path.parent / cube_filename

    try:
        config, exciton_data = IOHandler.read_cube(filename=str(cube_path),
                                                   shape_type=shape_type,
                                                   shape_params=shape_params)
    except FileNotFoundError:
        print(f"Critical Error: The cube file '{cube_path}' was not found.")
        sys.exit(1)

    print("\n--- Initializing Solver ---")
    solver = Solver(exciton_data, config, do_rdf_analysis=do_rdf)

    print("Running Physics Engine...")
    solver.solve()
    print("Analysis Complete.")

    print("Generating Mask Visualization...")
    mask = solver.build_visual_mask()

    debug_filename = f"{out_prefix}_MASK.cube"
    IOHandler.write_mask_cube(str(debug_filename), config, mask)

    # Output Results
    output_dir = input_path.parent
    txt_out = output_dir / f"{out_prefix}_OUT.txt"
    rdf_out = output_dir / f"{out_prefix}_1D-distance-involume.dat"

    IOHandler.write_report(str(txt_out), config, exciton_data)
    IOHandler.write_distance_involume(str(rdf_out), exciton_data)

    print(f"\n--- Done ---")
    print(f"Summary written to: {txt_out}")
    print(f"RDF Data written to: {rdf_out}")


    ### --- Graph Generation (Toggled by --print-analysis-graph) --- ###

    if args.print_analysis_graph:
        if not do_rdf:
            print(
                "\nWarning: Cannot generate analysis graph because In-Volume/RDF analysis is DISABLED in the input file.")
        else:
            print("\nGenerating Normalized Volume-Corrected Probability Density Graph...")

            r = exciton_data.rdf_distance
            prob_in = exciton_data.rdf_probability_in_vol
            prob_tot = exciton_data.rdf_probability_total
            counts = exciton_data.rdf_counts
            dv = config.dv

            valid = counts > 0
            raw_in_shell = prob_in / dv
            raw_tot_shell = prob_tot / dv

            raw_dens_in_shell = np.zeros_like(raw_in_shell)
            raw_dens_tot_shell = np.zeros_like(raw_tot_shell)

            raw_dens_in_shell[valid] = raw_in_shell[valid] / counts[valid]
            raw_dens_tot_shell[valid] = raw_tot_shell[valid] / counts[valid]

            cum_raw_in = np.cumsum(raw_dens_in_shell)
            cum_raw_tot = np.cumsum(raw_dens_tot_shell)

            # Scale to plateau at 1.0/dV
            expected_plateau_vol = 1.0 / dv
            plateau_raw = cum_raw_tot[-1] if cum_raw_tot[-1] > 0 else 1.0
            scale_factor_vol = expected_plateau_vol / plateau_raw

            cum_raw_in = cum_raw_in * scale_factor_vol
            cum_raw_tot = cum_raw_tot * scale_factor_vol

            fig, ax = plt.subplots(figsize=(8, 6))
            ax.plot(r, cum_raw_tot, 'k--', linewidth=1.5, label='Total')
            ax.plot(r, cum_raw_in, 'g-', linewidth=2.5, label='Inside Shape')
            ax.set_title(f"Volumetric Density (Volume-Corrected): {out_prefix}", fontsize=14)
            ax.set_xlabel("Distance [Bohr]", fontsize=12)
            ax.set_ylabel("Scaled Cumulative Density [Bohr$^{-3}$]", fontsize=12)
            ax.set_xlim(0, np.max(r))
            ax.grid(True, linestyle='--', alpha=0.6)
            ax.legend(loc='best')

            graph_out = output_dir / f"{out_prefix}_Volume_Corrected_Density.png"
            plt.tight_layout()
            plt.savefig(graph_out, dpi=300)
            print(f"  > Saved Analysis Graph to: {graph_out}")
            plt.close()


if __name__ == "__main__":
    main()