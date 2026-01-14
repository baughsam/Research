import argparse
import sys
from pathlib import Path
import numpy as np

# Import package modules
from ct_character.IOHandler import IOHandler
from ct_character.Solver import Solver
from ct_character.Exciton import ExcitonData, Configuration


def parse_input_file(filepath: Path):
    """
    Parses 6-line master input file format.
    Returns:
        cube_file (str): Path to the .cube file
        shape_type (str): 'EllipticalCylinder' or 'Parallelepiped'
        shape_params (dict): Dependent on shape_type
        output_prefix (str): Prefix for output files
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



        return cube_file, shape_type, shape_params, output_prefix

    except Exception as e:
        print(f"Error parsing input file '{filepath}'. {e}")
        sys.exit(1)


def main():
    # 1. Parse Command Line Arguments
    # This allows users to run: python main.py input.in
    parser = argparse. ArgumentParser(description="Charge Transfer Analysis Code")
    parser.add_argument("input_file", type=str, help="Path to the master input file (e.g., INPUT_CTCALC.in")
    args = parser.parse_args()

    input_path = Path(args.input_file)
    if not input_path.exists():
        print(f"Error: Input file '{input_path}' not found.")
        sys.exit(1)

    print(f"--- Starting CT analysis on {input_path} ---")

    # Parse Input File
    cube_filename, shape_type, shape_params, out_prefix = parse_input_file(input_path)

    print(f"  Target Cube File: {cube_filename}")
    print(f"  Shape Model:      {shape_type}")
    if shape_params:
        print(f"  Params Found:     {list(shape_params.keys())}")
    else:
        print(f"  Params Found:     NONE (Will use Class Defaults)")

    # Read Data
    # We assume the .cube file is in the same folder as the input file,
    # or we treat the path in the input file as relative to the execution location.
    # Here we resolve it relative to the input file's directory if it's not absolute.
    cube_path = Path(cube_filename)
    if not cube_path.is_absolute():
        cube_path = input_path.parent / cube_filename

    # IOHandler.read_cube returns the Config and ExcitonData objects
    try:
        config, exciton_data = IOHandler.read_cube(filename=str(cube_path),
                                                   shape_type=shape_type,
                                                   shape_params=shape_params)

    except FileNotFoundError:
        print(f"Critical Error: The cube file '{cube_path}' was not found.")
        sys.exit(1)

    # Initialize & Run Solver
    print("\n--- Initializing Solver ---")
    solver = Solver(exciton_data, config)

    print("Running Physics Engine...")
    solver.solve()
    print("Analysis Complete.")

    # Visual Shape
    # Re-generate the mask solely for visualization
    # (Since Solver uses it internally but doesn't store the boolean array)
    print("Generating Mask Visualization...")
    X, Y, Z, R = solver._generate_coordinates()
    mask = solver._create_volume_mask(X, Y, Z)

    debug_filename = f"{out_prefix}_MASK.cube"
    IOHandler.write_mask_cube(str(debug_filename), config, mask)
    # -----------------------

    # 5. Output Results
    # construct output filenames based on the prefix read from input
    # We save them in the same folder as the input file
    output_dir = input_path.parent
    txt_out = output_dir / f"{out_prefix}_OUT.txt"
    json_out = output_dir / f"{out_prefix}_stats.json"

    IOHandler.write_report(str(txt_out), config, exciton_data)
    IOHandler.write_report(str(json_out), config, exciton_data)

    print(f"\n--- Done ---")
    print(f"Summary written to: {txt_out}")

if __name__ == "__main__":
    main()