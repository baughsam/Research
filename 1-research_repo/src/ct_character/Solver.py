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

        # Calculate Multipoles (Dipole & Quadrupole)
        self._calculate_multipoles(X, Y, Z)

        # Calculate 1D & 2D Projections (Averaging)
        self._calculate_projections(X, Y, Z)


"""
    def _generate_coordinates(self):
        Generates 3D Cartesian coordinates (X, Y, Z) and Radius (R).
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
"""


    def _generate_coordinates(self):
        """
        Generates 3D Cartesian coordinates (X, Y, Z) and Radius (R).
        OPTIMIZED: Uses float32 and in-place math to prevent MemoryError.
        """
        nx, ny, nz = self.data.grid_data.shape

        # 1. Use float32 indices (Saves 50% RAM compared to float64)
        i, j, k = np.indices((nx, ny, nz), dtype=np.float32)

        # 2. Center indices in-place (No new array creation)
        i -= (nx / 2.0)
        j -= (ny / 2.0)
        k -= (nz / 2.0)

        # 3. Manual Transform (Avoids creating giant 'stack' or 'tensordot' copies)
        M = self.config.transform_matrix

        # Allocate X, Y, Z directly from the linear combinations
        # X = i*M[0,0] + j*M[0,1] + k*M[0,2]
        X = i * M[0, 0] + j * M[0, 1] + k * M[0, 2]
        Y = i * M[1, 0] + j * M[1, 1] + k * M[1, 2]
        Z = i * M[2, 0] + j * M[2, 1] + k * M[2, 2]

        # 4. CRITICAL: Delete indices immediately to free ~2-3 GB
        del i, j, k

        # 5. Calculate R with memory-safe accumulation
        # Don't use R = np.sqrt(X**2 + Y**2 + Z**2) because it creates 3 huge temp arrays.

        R = np.square(X)  # R holds X^2
        R += np.square(Y)  # Add Y^2 in-place
        R += np.square(Z)  # Add Z^2 in-place
        np.sqrt(R, out=R)  # Square root in-place

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
            return 1 - ( masked_sum / total_density )
        return 0.0

    def _calculate_rdf_and_avg_radius(self, R):
        """
        Calculates Radial Distribution Function and <r>
        """

        vecs = self.config.lattice_vectors
        box_diagonal_vector = vecs[0] + vecs[1] + vecs[2]
        diagonal = np.linalg.norm(box_diagonal_vector)

        # ... inside _calculate_rdf_and_avg_radius ...

        # --- Replicate Fortran "Loop" Logic for Nb_Distances ---
        # Fortran: Iterates id=1..N. IF (Diag/id > Max_Voxel) Nb=id.
        # This finds the largest Number of Bins where the Step Size is still > Voxel Size.

        # Calculating voxel size
        step_vectors = np.linalg.norm(self.config.transform_matrix, axis=0)
        max_voxel_step = np.max(step_vectors)

        nb_bins = 1
        # We simulate the Fortran loop up to a reasonable max (e.g. total voxels)
        # Only go as high as needed (Diag / Voxel)
        limit = int(diagonal / max_voxel_step) + 5

        for i in range(1, limit):
            current_step = diagonal / i
            if current_step > max_voxel_step:
                nb_bins = i
            # Fortran continues looping, but Nb only updates if condition is met.
        """
        # Calculating voxel size
        step_vectors = np.linalg.norm(self.config.transform_matrix, axis=0)
        max_voxel_step = np.max(step_vectors) # 'Max Norm'

        # Find Nb_Distance
        # "IF (DistanceStep > Maximum_norm) Nb_Distances=id"
        # Algebraic Interpretation of the loop:
        # diagonal / nb_bins > max_voxel_step -> nb_bins < diagonal / max_voxel_step
        nb_bins = int(diagonal / max_voxel_step)
        """

        # Create Bins
        bins = np.linspace(0, diagonal, nb_bins + 1)


        # Total Density Histogram
        hist_total_rho, bin_edges = np.histogram(R, bins=bins, weights=self.data.grid_data)

        # In-Volume Density
        hist_in_vol_rho, _ = np.histogram(R, bins=bins, weights=self.data.density_inside_shape)

        # Volume Count (Shell Volume in Voxels)
        hist_counts, _ = np.histogram(R, bins=bins)

        #Store raw counts for output
        self.data.rdf_counts = hist_counts


        # Bin centers
        # Can't plot data point at a "boundary". we have to plot it at the center
        #self.data.rdf_distance = (bin_edges[:-1] + bin_edges[1:]) / 2.0

        # Bin (Left Edge)
        self.data.rdf_distance = bin_edges[:-1]


        # Safe division for avg density at r
        with (np.errstate(divide='ignore', invalid='ignore')): # Empty bins won't crash Python (division by zero)
            avg_rho_total = np.nan_to_num(hist_total_rho / hist_counts)
            avg_rho_in_vol = np.nan_to_num(hist_in_vol_rho / hist_counts)

        # Normalization
        cdf_total = np.sum(avg_rho_total)

        if cdf_total > 1e-12:
            self.data.rdf_values = avg_rho_total / cdf_total
            self.data.rdf_in_volume_values = avg_rho_in_vol / cdf_total
        else:
            self.data.rdf_values = avg_rho_total
            self.data.rdf_in_volume_values = avg_rho_in_vol

        # Store raw density profile if needed for other cals
        self.data.density_distance = avg_rho_total

        # Average Radius of Electron <r>
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
        self.data.avg_r = np.sum(self.data.rdf_values * bin_centers)

    def _calculate_multipoles(self, X, Y, Z):
        """
        Calculates Dipole and Quadrupole moments using masked density.
        """
        rho = self.data.density_inside_shape
        total_dens = np.sum(self.data.grid_data)
        norm = 1.0 / total_dens if total_dens > 1e-12 else 0.0

        # Normalized masked density
        rho_norm = rho * norm

        # Dipole
        self.data.dipole_moment = np.array([
            np.sum(rho_norm * X),
            np.sum(rho_norm * Y),
            np.sum(rho_norm * Z)
        ])

        # Quadrupole (Symmetric Tensor)
        R2 = X**2 + Y**2 + Z**2

        qxx = np.sum(rho_norm * (3*X**2 - R2))
        qyy = np.sum(rho_norm * (3*Y**2 - R2))
        qzz = np.sum(rho_norm * (3*Z**2 - R2))
        qxy = np.sum(rho_norm * (3*X*Y))
        qxz = np.sum(rho_norm * (3*X*Z))
        qyz = np.sum(rho_norm * (3*Y*Z))

        self.data.quadrupole_moment = np.array([
            [qxx, qxy, qxz],
            [qxy, qyy, qyz],
            [qxz, qyz, qzz]
        ])

    def _calculate_projections(self, X, Y, Z):
        """
        1-D Planar Averages
        Calculates 1st (<|x|>) and 2nd (<x^2>) moments projected onto lattice vectors.
        """
        rho = self.data.density_inside_shape
        total_dens = np.sum(self.data.grid_data)

        # Save total weight for the report
        self.data.total_weight = total_dens * self.config.dv

        # Normalize density (Probability Distribution)
        norm = 1.0 / total_dens if total_dens > 1e-12 else 0.0
        rho_norm = rho * norm

        # --- Get Lattice Unit Vectors ---
        # We need directions: a_hat, b_hat, c_hat
        vecs = self.config.lattice_vectors  # [a, b, c]

        # Calculate norms (lengths) of lattice vectors
        a_len = np.linalg.norm(vecs[0])
        b_len = np.linalg.norm(vecs[1])
        c_len = np.linalg.norm(vecs[2])

        # Create Unit Vectors (Direction only)
        a_hat = vecs[0] / a_len
        b_hat = vecs[1] / b_len
        c_hat = vecs[2] / c_len

        # --- Project Cartesian (X,Y,Z) onto Lattice Directions ---
        # Formula: Proj = X*ux + Y*uy + Z*uz (Dot Product)
        # Note: X, Y, Z are 3D arrays. a_hat is a vector.
        proj_a = X * a_hat[0] + Y * a_hat[1] + Z * a_hat[2]
        proj_b = X * b_hat[0] + Y * b_hat[1] + Z * b_hat[2]
        proj_c = X * c_hat[0] + Y * c_hat[1] + Z * c_hat[2]

        # Radial Distance (already calculated in generate_coordinates, but R is passed in usually)
        # We can re-calculate R locally if needed, or rely on X,Y,Z
        R = np.sqrt(X ** 2 + Y ** 2 + Z ** 2)

        # --- Calculate First Moments <|x|> ---
        # Sum ( Density * Abs(Distance) )
        self.data.avg_a = np.sum(rho_norm * np.abs(proj_a))
        self.data.avg_b = np.sum(rho_norm * np.abs(proj_b))
        self.data.avg_c = np.sum(rho_norm * np.abs(proj_c))
        self.data.avg_r = np.sum(rho_norm * R)

        # --- 4. Calculate Second Moments <x^2> ---
        # Sum ( Density * Distance^2 )
        self.data.avg_a2 = np.sum(rho_norm * (proj_a ** 2))
        self.data.avg_b2 = np.sum(rho_norm * (proj_b ** 2))
        self.data.avg_c2 = np.sum(rho_norm * (proj_c ** 2))
        self.data.avg_r2 = np.sum(rho_norm * (R ** 2))