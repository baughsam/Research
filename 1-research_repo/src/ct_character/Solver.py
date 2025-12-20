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

        # Create boolean (geometric) mask
        mask = self._create_volume_mask(X, Y, Z)

        # Apply boolean mask to density
        self._apply_mask_to_density(mask)

        # Calculate CT Ratio
        self.data.ct_ratio = self._calculate_ct_ratio()

        # Calculate Radial Distribution (RDF) & Average Radius
        self._calculate_rdf_and_avg_radius(R)



    def _generate_coordinates(self):
        """Generates 3D Cartesian coordinates (X, Y, Z) and Radius (R)."""
        nx, ny, nz = self.data.grid_data.shape
        i, j, k = np.indices((nx,ny,nz), dtype=float)

        # Center indices (0,0,0 at center of box)
        di = i - (nx / 2.0)
        dj = j - (ny / 2.0)
        dk = k - (nz / 2.0)

        #Stack for matrix multiplication
        grid_coords = np.stack([di, dj, dk], axis=0)

        # Transform Grid -> Cartesian using Lattice Matrix
        # shape: (3, nx, ny, nz)
        coords = np.tensordot(self.config.transform_matrix, grid_coords, axes=(1,0))

        X, Y, Z = coords[0], coords[1], coords[2]
        R = np.sqrt(X**2 + Y**2 + Z**2)

        return X, Y, Z, R

    def _create_volume_mask(self, X, Y, Z):
        """
        Asks the Shape object to create boolean mask.
        """

        # Shape object from Configuration
        shape_obj = self.config.shape

        # Ask the shape: "Which of these x amount of points are inside?"
        # Returns an array of (nx, ny, nz) values with either True or False
        mask = shape_obj.is_inside(X, Y, Z)

        return mask

    def _apply_mask_to_density(self, mask):
        """
        Filters the density grid using the mask.
        """
        self.data.density_inside_shape = np.where(mask, self.data.grid_data, 0.0)

    def _calculate_ct_ratio(self):
        """
        Calculates fraction of charge inside Shape volume
        """
        total_density = np.sum(self.data.grid_data)
        masked_sum = np.sum(self.data.density_inside_shape)

        if total_density > 1e-12:
            return masked_sum / total_density
        return 0.0

    def _calculate_rdf_and_avg_radius(self, R):
        """
        Calculates Radial Distribution Function and <r>
        """
        # Binning
        nb_bins = min(300, int(np.max(self.data.grid_data.shape)))
        bins = np.linspace(0, np.max(R), nb_bins +1)

        # Weighted histogram (sum of density in shell)
        hist_rho, bin_edges = np.histogram(R, bins=bins, weights=self.data.grid_data)
        # Count histogram (volume of shell in voxels)
        hist_counts, _ = np.histogram(R, bins=bins)

        #Bin centers
        # Can't ploat data point at a "boundary". we have to plot it at the center
        self.data.rdf_distance = (bin_edges[:-1] + bin_edges[1:]) / 2.0

        # Safe division for avg density at r
        with np.errstate(divide='ignore', invalid='ignore'): # Empty bins won't crash Python (division by zero)
            avg_rho = np.nan_to_num(hist_rho / hist_counts) #Converts NaN to zero

        # Raw (non-normalized) density profile
        self.data.density_distance = avg_rho # Raw profile

        #Normalized RDF (CDF = 1)
        cdf = np.cumsum(avg_rho)
        if cdf[-1] > 0:
            self.data.rdf_values = avg_rho / cdf[-1]
        else:
            self.data.rdf_values = avg_rho

        # Average Radius of Electron from Center <r>
        # Sum (Probability of being at distance r * Distance)
        self.data.avg_r = np.sum(self.data.rdf_values * self.data.rdf_distance)