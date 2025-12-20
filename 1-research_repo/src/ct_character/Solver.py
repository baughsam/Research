import numpy as np

class Solver:
    def __init__(self, exciton_data, configuration):
        """
        Initializes Solver w/ ExcitonData and Configuration

        Args:
            exciton_data: An object containing the 3D density grid and empty placeholder
                          for results (ExcitonData class).
            configuration: An object containing input parameters like cell dimensions,
            lattice vectors, and shape type (Configuration class).
        """
        self.data = exciton_data
        self.config = configuration

    def solve(self):
        """
        Conductor method.
        It orchestrates the analysis, delegating the actual math to helper methods
        """

        # Generate the coordinate grids
        X, Y, Z, R = self._generate_coordinates()

    def _generate_coordinates(self):
        """Generates 3D Cartesian coordinates (X, Y, Z) and Radius (R)."""
        nx, ny, nz = self.data.grid_data.shape
        i, j, k = np.indices((nx,ny,nz), dtype=float)

        # Center indices (0,0,0 at center of box)
        di = i - (nx / 2.0)
        dj = j - (ny / 2.0)
        dk = k - (nz / 2.0)

        #Stack for matrix multiplicaiton
        grid_coords = np.stack([di, dj, dk], axis=0)

        # Transform Grid -> Cartesian using Lattice Matrix
        # shape: (3, nx, ny, nz)
        coords = np.tensordot(self.config.transform_matrix, grid_coords, axes=(1,0))

        X, Y, Z = coords[0], coords[1], coords[2]
        R = np.sqrt(X**2 + Y**2 + Z**2)

        return X, Y, Z, R
