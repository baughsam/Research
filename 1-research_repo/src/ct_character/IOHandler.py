import numpy as np
import Exciton
from ct_character.Exciton import Configuration, ExcitonData
from ct_character.Shape import Shape
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
