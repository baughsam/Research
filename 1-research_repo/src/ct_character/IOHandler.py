import numpy as np
import Exciton
from ct_character.Exciton import Configuration, ExcitonData
from ct_character.Shape import EllipticalCylinder
from pathlib import Path
import os

class IOHandler:

    @staticmethod
    def read_cube(filename: str, shape_parms: dict) -> tuple[Exciton.Configuration, Exciton.ExcitonData]:
        """
        Extracts Configuration and ExcitonData from a .cube file
        """
        file_path = Path(filename)
        print(f"Reading metadata from: {file_path}...")

        # Metadata
        with open(file_path, "r") as f:
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

            if cache_path.exists():
                print(f"Found binary cache. Loading fast from: {cache_path}")
                # mmap_mod='r' allows us to read > 1GB files without instantly filling RAM
                density = np.load(cache_path, mmap_mode='r')

                # Sanity Check: Does the cached file match the header we just read?
                if density.shape != tuple(grid_shape):
                    print(f"Warning: Cache shape mismatch. Regenerating density...")
                    density = None
                else:
                    print("No cache found.")
                    density = None

                if density is None:
                    print("Parsing text volumatric data (Slow)...")

                    raw_data = np.fromstring(f.read(), sep=' ')
                    density = raw_data.reshape(tuple(grid_shape))

                    print(f"Saving binary cache to: {cache_path}")
                    np.save(cache_path, density)
        # --- Build Objects --- #
        specific_shape = EllipticalCylinder(**shape_parms)
        config = Configuration(
            lattice_vectors=lattice_vectors,
            origin=origin,
            grid_shape=tuple(grid_shape),
            shape= specific_shape,
            atom_types=atom_types,
            atom_positions=atom_positions,
        )

        data = ExcitonData(density=density)

        return config, data

    @staticmethod
    def write_report(filename: str, config: Exciton.Configuration, data: Exciton.ExcitonData):
        """
        Writes comprehensive summary file (_OUT.txt in .f90)
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

            # --- CT & DIPOLE RESULTS --- #
            f.write("Key Results:\n")
            if data.ct_ratio is not None:
                f.write(f" Charge Transfer Ratio: {data.ct_ratio:.6f}\n")

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