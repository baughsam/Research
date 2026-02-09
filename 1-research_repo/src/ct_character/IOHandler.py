import numpy as np
from ct_character.Exciton import Configuration, ExcitonData
from ct_character.Shape import EllipticalCylinder
import ct_character.Shape as ShapeModule
from pathlib import Path
import datetime

class IOHandler:

    @staticmethod
    def read_cube(filename: str, shape_params: dict, shape_type: str) -> tuple[Configuration, ExcitonData]:
        """
        Extracts Configuration and ExcitonData from a .cube file
        """
        file_path = Path(filename)
        print(f"Reading metadata from: {file_path}...")

        # Metadata
        with (open(file_path, "r") as f):
            # Skip Comments
            f.readline(); f.readline()

            # --- Origin & Atoms --- #
            line3 = f.readline().split()
            n_atoms = int(line3[0])
            origin = np.array([float(x) for x in line3[1:4]])

            # --- Grid and Lattice --- #
            grid_shape = []
            step_vectors = []
            for _ in range(3):
                line = f.readline().split()
                grid_shape.append(int(line[0]))
                step_vectors.append([float(line[1]), float(line[2]), float(line[3])])

            grid_shape = tuple(grid_shape)
            step_vectors = np.array(step_vectors)

            # --- Calculating Lattice Vectors --- #
            # Lattice = Step_Vector * Number_of_Voxels on that axis
            lattice_vectors = np.zeros((3, 3))
            for i in range(3):
                lattice_vectors[i] = step_vectors[i] * grid_shape[i]

            # --- Parsing Atoms --- #
            atom_positions = []
            atom_types = []

            for _ in range(abs(n_atoms)):
                line = f.readline().split()
                atom_types.append(int(line[0]))
                atom_positions.append([float(line[2]), float(line[3]), float(line[4])])

            atom_positions = np.array(atom_positions)
            atom_types = np.array(atom_types)

            # --- Volumetric Data Parsing --- #
            # Using a binary data parser for bigger .cube files.
            # First uses text parser to make and save a .npy file
            # On subsequent runs it will pick the .npy file

            cache_path = file_path.with_name(f"{file_path.stem}_density.npy")

            # Initialize Density
            density = None

            if cache_path.exists():
                print(f"Found binary cache. Loading fast from: {cache_path}")
                try:
                    # mmap_mod='r' allows us to read > 1GB files without instantly filling RAM
                    loaded_density = np.load(cache_path, mmap_mode='r')

                    # Sanity Check: Does the cached file match the header we just read?
                    if density.shape != tuple(grid_shape):
                        print(f"Warning: Cache shape mismatch. Regenerating density...")
                        density = None
                    else:
                        density = loaded_density

                except Exception as e:
                    print(f"Warning: Failed ot load cache: {e}")
            else:
                print("No cache found")

            if density is None:
                print("Parsing text volumatric data (Slow)...")

                # Fixes Fortran 'D' notation (just in case)
                content = f.read().replace('D', 'E').replace('d', 'E')

                raw_data = np.fromstring(content, sep=' ').astype(np.float32)
                density = raw_data.reshape(tuple(grid_shape))


                print(f"Saving binary cache to: {cache_path}")
                np.save(cache_path, density)

        # --- Build Objects --- #
        try:
            # Look for a class in Shape.py that matches the string
            selected_class = getattr(ShapeModule, shape_type)
            print(f"using Shape Class: {shape_type}")

        except AttributeError:
            # If class name doesn't exist
            print(f"Warning: Shape class '{shape_type}' not found in Shape.py.\n")
            print("Defaulting to EllipticalCylinder.")
            selected_class = EllipticalCylinder

        try:
            specific_shape = selected_class(**shape_params)
        except TypeError as e:
            # Happens if shape_params is empty or has wrong variables/keys
            # This happens if shape_params is empty OR has the wrong keys
            print(f"Warning: Could not instantiate {shape_type} with provided params.")
            print(f"  Error Detail: {e}")
            print(f"  -> Using DEFAULT values for {shape_type}")
            specific_shape = selected_class()


        config = Configuration(
            lattice_vectors=lattice_vectors,
            origin=origin,
            grid_shape=tuple(grid_shape),
            shape= specific_shape,
            atom_types=atom_types,
            atom_positions=atom_positions,
        )

        data = ExcitonData(grid_data=density)

        return config, data

    @staticmethod
    def write_report(filename: str, config: Configuration, data: ExcitonData):
        """
        Writes comprehensive summary file (Simplified for CT Ratio Only)
        """
        print(f"Writing summary to: {filename}...")

        # Get current time
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        with open(filename, 'w') as f:
            # --- HEADER ---
            f.write("*************************************************\n")
            f.write("           Exciton Analysis Summary              \n")
            f.write("*************************************************\n")
            f.write(f"  Date: {now}\n")  # <--- NEW LINE
            f.write("*************************************************\n\n")

            # --- System Parameters ---
            nx, ny, nz = config.grid_shape
            f.write(f"  Grid dimensions:       {nx} x {ny} x {nz}\n")
            f.write(f"  Cell volume:           {config.total_volume:.6f} Bohr^3\n")
            f.write(f"  Voxel volume (dV):     {config.dv:.6e} Bohr^3\n")
            f.write(f"  Shape Model:           {type(config.shape).__name__}\n\n")

            # --- CT & Wfn Norm ---
            f.write("Key Results:\n")
            f.write(f" Wavefunction Norm:      {data.total_weight:.6e} electrons\n")
            if data.ct_ratio is not None:
                f.write(f" Charge Transfer Ratio:  {data.ct_ratio:.6f}\n")

    @staticmethod
    def write_mask_cube(filename: str, config: Configuration, mask: np.ndarray):
        """
        Debug Tool: Writes the boolean mask to a .cube file.
        Updated to strictly follow Gaussian Cube format (newlines per Z-row).
        """
        print(f"DEBUG: Writing mask visualization to {filename}...")

        nx, ny, nz = config.grid_shape
        # Convert True/False -> 1.0/0.0
        mask_data = mask.astype(float)

        with open(filename, 'w') as f:
            # --- Header ---
            f.write("Mask Debug File\n")
            f.write("Generated by ct_character\n")

            # --- Origin & Atoms ---
            # IMPORTANT: Negative number of atoms tells VESTA/VASP that units are BOHR
            n_atoms_flag = -len(config.atom_types)
            f.write(f"{n_atoms_flag:5d} {config.origin[0]:12.6f} {config.origin[1]:12.6f} {config.origin[2]:12.6f}\n")

            # --- Lattice Vectors ---
            vecs = config.lattice_vectors
            step_x = vecs[0] / nx
            step_y = vecs[1] / ny
            step_z = vecs[2] / nz

            f.write(f"{nx:5d} {step_x[0]:12.6f} {step_x[1]:12.6f} {step_x[2]:12.6f}\n")
            f.write(f"{ny:5d} {step_y[0]:12.6f} {step_y[1]:12.6f} {step_y[2]:12.6f}\n")
            f.write(f"{nz:5d} {step_z[0]:12.6f} {step_z[1]:12.6f} {step_z[2]:12.6f}\n")

            # --- Atoms ---
            for i in range(len(config.atom_types)):
                pos = config.atom_positions[i]
                # AtomType Charge X Y Z
                f.write(f"{config.atom_types[i]:5d} {0.0:12.6f} {pos[0]:12.6f} {pos[1]:12.6f} {pos[2]:12.6f}\n")

            # --- Volumetric Data (Strict Loop) ---
            # Gaussian Format: 6 values per line.
            # CRITICAL: Must start a new line at the start of every Z-row (ix, iy).

            for ix in range(nx):
                for iy in range(ny):
                    line_buffer = []
                    for iz in range(nz):
                        val = mask_data[ix, iy, iz]
                        line_buffer.append(f"{val:13.5E}")

                        # Flush every 6 values
                        if len(line_buffer) == 6:
                            f.write("".join(line_buffer) + "\n")
                            line_buffer = []

                    # Flush remaining values at the end of the Z-row
                    if line_buffer:
                        f.write("".join(line_buffer) + "\n")

    @staticmethod
    def write_distance_involume(filename: str, data: ExcitonData):
        """
        Writes RDF analysis with BOTH Legacy (Density) and Correct (Probability) columns.
        """
        print(f"Writing RDF analysis to: {filename}...")

        if data.rdf_distance is None:
            print("Warning: RDF data missing.")
            return

        with open(filename, 'w') as f:
            # Clear Header explaining the columns
            f.write(f"# {'Dist':>12} {'Legacy_In':>14} {'Legacy_Tot':>14} {'Prob_In':>14} {'Prob_Tot':>14}\n")
            f.write(f"# {'[Bohr]':>12} {'(Avg Rho)':>14} {'(Avg Rho)':>14} {'(Norm)':>14} {'(Norm)':>14}\n")

            for i in range(len(data.rdf_distance)):
                dist = data.rdf_distance[i]

                # Legacy (Density)
                leg_in = data.rdf_density_in_vol[i]
                leg_tot = data.rdf_density_total[i]

                # Correct (Probability)
                prob_in = data.rdf_probability_in_vol[i]
                prob_tot = data.rdf_probability_total[i]

                f.write(f"{dist:13.5E} {leg_in:13.5E} {leg_tot:13.5E} {prob_in:13.5E} {prob_tot:13.5E}\n")