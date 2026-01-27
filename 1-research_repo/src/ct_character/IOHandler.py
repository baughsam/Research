import numpy as np
from ct_character.Exciton import Configuration, ExcitonData
from ct_character.Shape import EllipticalCylinder
import ct_character.Shape as ShapeModule
from pathlib import Path
import json
import os

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

                raw_data = np.fromstring(f.read(), sep=' ')
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
        Writes comprehensive summary file
        """

        print(f"Writing summary to: {filename}...")

        with open(filename, 'w') as f:
            # --- HEADER ---
            f.write("*************************************************\n")
            f.write("           Exciton Analysis Summary              \n")
            f.write("*************************************************\n\n")

            # --- System Parameters --- *
            nx, ny, nz = config.grid_shape
            f.write(f"  Grid dimensions:       {nx} x {ny} x {nz}\n")
            f.write(f"  Cell volume:           {config.total_volume:.6f} Bohr^3\n")
            f.write(f"  Voxel volume (dV):     {config.dv:.6e} Bohr^3\n")
            f.write(f"  Shape Model:           {type(config.shape).__name__}\n\n")

            # --- CT & Wfn Norm --- #
            f.write("Key Results:\n")
            f.write(f" Wavefunction Norm:      {data.total_weight:.6e} Bohr^3\n")
            if data.ct_ratio is not None:
                f.write(f" Charge Transfer Ratio:  {data.ct_ratio:.6f}\n")

            # --- Dipole --- #
            if data.dipole_moment is not None:
                # Calculate magnitude of dipole
                dipole_mag = np.linalg.norm(data.dipole_moment)
                f.write(f" Dipole Moment (Vector): {data.dipole_moment}\n")
                f.write(f" Dipole Magnitude:       {dipole_mag:.6f} Bohr\n\n")

            # -- MOMENTS --- #
            if data.avg_r is not None:
                f.write("First Moment Metrics (Average Distance):\n")
                f.write(f"  <|r|> (Mean Radius):    {data.avg_r:.6f} Bohr\n")
                f.write(f"  <|a|> (Proj. on A):     {data.avg_a:.6f} Bohr\n")
                f.write(f"  <|b|> (Proj. on B):     {data.avg_b:.6f} Bohr\n")
                f.write(f"  <|c|> (Proj. on C):     {data.avg_c:.6f} Bohr\n\n")

            if data.avg_r2 is not None:
                f.write("Second Moment Metrics:\n")
                f.write(f"  <|r^2|>:     {data.avg_r:.6f} Bohr^2\n")
                f.write(f"  <|a^2|>:     {data.avg_a:.6f} Bohr^2\n")
                f.write(f"  <|b^2|>:     {data.avg_b:.6f} Bohr^2\n")
                f.write(f"  <|c^2|>:     {data.avg_c:.6f} Bohr^2\n\n")


            # --- ANISOTROPY --- #
            if data.avg_a is not None and data.avg_b is not None:
                ratio_ab = data.avg_a / data.avg_b if data.avg_a > 0 else 0
                f.write("Anisotropy:\n")
                f.write(f"  Ratio <|a|>/<|b|>:      {ratio_ab:.4f}\n")

    @staticmethod
    def write_json_stats(filename, config, data):
        """
        Writes JSON file w/ physical parameters for plotting purposes
        """
        stats = {
            "ct_ratio": data.ct_ratio,
            "dipole_magnitude": np.linalg.norm(data.dipole_moment),
            "dipole_vector": data.dipole_moment.tolist(),  # JSON can't handle numpy arrays
            "avg_radius": data.avg_r,
            "anisotropy_ab": data.avg_a / data.avg_b
        }
        with open(filename.replace(".txt", ".json"), 'w') as f:
            json.dump(stats, f, indent=4)

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
        Writes the '1D-distance-involume' file matching Fortran output.
        Columns: Distance, In-Volume-Rho, Total-Rho, Count
        """
        print(f"Writing RDF analysis to: {filename}...")

        # Ensure data exists
        if data.rdf_distance is None or data.rdf_in_volume_values is None:
            print("Warning: RDF data missing. Skipping output.")
            return

        with open(filename, 'w') as f:
            # Header matching Fortran style
            f.write("# Distance [Bohr], 2-part corr (In-Vol), 2-part corr (Total), NbPoints\n")

            # Iterate through bins
            for i in range(len(data.rdf_distance)):
                dist = data.rdf_distance[i]
                rho_in = data.rdf_in_volume_values[i]
                rho_tot = data.rdf_values[i]
                count = int(data.rdf_counts[i])

                # Fortran Format: 3ES13.5, I10
                # Python f-string equivalent: {val:13.5E}
                f.write(f"{dist:13.5E} {rho_in:13.5E} {rho_tot:13.5E} {count:10d}\n")