import numpy as np


class Solver:
    def __init__(self, exciton_data, configuration):
        self.data = exciton_data
        self.config = configuration

    def solve(self):
        """
        Conductor method.
        Orchestrates the analysis using Memory-Optimized Z-Plane Slicing.
        """
        print("Running Memory-Optimized Solver (Z-Plane Slicing)...")

        nx, ny, nz = self.data.grid_data.shape

        # 1. Initialize Accumulators (Holds running totals for all methods)
        acc = self._initialize_accumulators()

        # 2. Pre-calculate the XY Grid (Constant for all Z-planes)
        #    This prevents regenerating the same X and Y coordinates 500 times.
        xy_base_coords = self._precompute_xy_grid(nx, ny)
        z_step_vec = self.config.transform_matrix[:, 2]

        # 3. Z-Plane Loop (The Memory Saver)
        for k in range(nz):
            # A. Get Data for this slice
            rho_slice = self.data.grid_data[:, :, k]

            # B. Generate Coordinates for this Z-plane
            X, Y, Z, R = self._generate_slice_coords(k, nz, xy_base_coords, z_step_vec)

            # C. Create Mask for this slice
            mask_slice = self._create_volume_mask(X, Y, Z)

            # D. Apply Mask
            rho_inside = np.where(mask_slice, rho_slice, 0.0)

            # --- DELEGATE TO HELPER METHODS (Accumulation Phase) ---
            self._accumulate_ct_ratio(acc, rho_slice, rho_inside)
            self._accumulate_rdf(acc, R, rho_slice, rho_inside)
            self._accumulate_multipoles(acc, X, Y, Z, R, rho_inside)
            self._accumulate_projections(acc, X, Y, Z, R, rho_inside)

        # 4. Finalize (Normalize sums and store in self.data)
        self._finalize_results(acc)
        print("Solver finished successfully.")

    def _initialize_accumulators(self):
        """Prepares zeroed variables to hold sums during the loop."""
        # RDF Binning Setup
        vecs = self.config.lattice_vectors
        box_diag = np.linalg.norm(vecs[0] + vecs[1] + vecs[2])
        step_vecs = np.linalg.norm(self.config.transform_matrix, axis=0)
        max_step = np.max(step_vecs)

        nb_bins = int(box_diag / max_step)
        bins = np.linspace(0, box_diag, nb_bins + 1)

        return {
            'total_density': 0.0,
            'masked_density': 0.0,
            'dipole': np.zeros(3),
            'quadrupole': np.zeros((3, 3)),
            # Exact Moments
            'moment_r': 0.0, 'moment_a': 0.0, 'moment_b': 0.0, 'moment_c': 0.0,
            'moment_r2': 0.0, 'moment_a2': 0.0, 'moment_b2': 0.0, 'moment_c2': 0.0,
            # RDF Data
            'bins': bins,
            'hist_counts': np.zeros(nb_bins),
            'hist_total_mass': np.zeros(nb_bins),
            'hist_in_vol_mass': np.zeros(nb_bins),
            # Projections vectors
            'a_hat': vecs[0] / np.linalg.norm(vecs[0]),
            'b_hat': vecs[1] / np.linalg.norm(vecs[1]),
            'c_hat': vecs[2] / np.linalg.norm(vecs[2])
        }

    def _precompute_xy_grid(self, nx, ny):
        """Generates the X and Y components of the grid once."""
        i_2d, j_2d = np.indices((nx, ny), dtype=np.float32)
        di = i_2d - ((nx / 2.0) - 1)
        dj = j_2d - ((ny / 2.0) - 1)
        matrix = self.config.transform_matrix

        # r_xy = i * vec_a + j * vec_b
        xy_coords = (
                np.outer(matrix[:, 0], di.ravel()).reshape(3, nx, ny) +
                np.outer(matrix[:, 1], dj.ravel()).reshape(3, nx, ny)
        )
        return xy_coords

    def _generate_slice_coords(self, k, nz, xy_base, z_step_vec):
        """Adds the Z-component to the precomputed XY grid for the current slice."""
        dk = float(k) - ((nz / 2.0) - 1)
        current_coords = xy_base + z_step_vec[:, np.newaxis, np.newaxis] * dk

        X = current_coords[0]
        Y = current_coords[1]
        Z = current_coords[2]
        R = np.sqrt(X ** 2 + Y ** 2 + Z ** 2)
        return X, Y, Z, R

    def _create_volume_mask(self, X, Y, Z):
        """Asks the Shape object to create boolean mask for this slice."""
        return self.config.shape.is_inside(X, Y, Z)

    # --- ACCUMULATION METHODS (Replaces your original calculation methods) ---

    def _accumulate_ct_ratio(self, acc, rho_slice, rho_inside):
        """Updates density running totals."""
        acc['total_density'] += np.sum(rho_slice)
        acc['masked_density'] += np.sum(rho_inside)

    def _accumulate_rdf(self, acc, R, rho_slice, rho_inside):
        """Updates RDF histograms for both Legacy and Exact profiles."""
        bins = acc['bins']
        counts, _ = np.histogram(R, bins=bins)
        mass_tot, _ = np.histogram(R, bins=bins, weights=rho_slice)
        mass_in, _ = np.histogram(R, bins=bins, weights=rho_inside)

        acc['hist_counts'] += counts
        acc['hist_total_mass'] += mass_tot
        acc['hist_in_vol_mass'] += mass_in

    def _accumulate_multipoles(self, acc, X, Y, Z, R, rho_inside):
        """Updates Dipole and Quadrupole running sums."""
        acc['dipole'][0] += np.sum(rho_inside * X)
        acc['dipole'][1] += np.sum(rho_inside * Y)
        acc['dipole'][2] += np.sum(rho_inside * Z)

        R2 = R ** 2
        q = acc['quadrupole']
        q[0, 0] += np.sum(rho_inside * (3 * X ** 2 - R2))
        q[1, 1] += np.sum(rho_inside * (3 * Y ** 2 - R2))
        q[2, 2] += np.sum(rho_inside * (3 * Z ** 2 - R2))
        q[0, 1] += np.sum(rho_inside * (3 * X * Y))
        q[0, 2] += np.sum(rho_inside * (3 * X * Z))
        q[1, 2] += np.sum(rho_inside * (3 * Y * Z))

    def _accumulate_projections(self, acc, X, Y, Z, R, rho_inside):
        """
        Calculates Exact 3D Moments (<|x|>, <x^2>) via grid summation.
        Replaces your original _calculate_projections.
        """
        proj_a = X * acc['a_hat'][0] + Y * acc['a_hat'][1] + Z * acc['a_hat'][2]
        proj_b = X * acc['b_hat'][0] + Y * acc['b_hat'][1] + Z * acc['b_hat'][2]
        proj_c = X * acc['c_hat'][0] + Y * acc['c_hat'][1] + Z * acc['c_hat'][2]

        # 1st Moments
        acc['moment_r'] += np.sum(rho_inside * R)
        acc['moment_a'] += np.sum(rho_inside * np.abs(proj_a))
        acc['moment_b'] += np.sum(rho_inside * np.abs(proj_b))
        acc['moment_c'] += np.sum(rho_inside * np.abs(proj_c))

        # 2nd Moments
        acc['moment_r2'] += np.sum(rho_inside * R ** 2)
        acc['moment_a2'] += np.sum(rho_inside * proj_a ** 2)
        acc['moment_b2'] += np.sum(rho_inside * proj_b ** 2)
        acc['moment_c2'] += np.sum(rho_inside * proj_c ** 2)

    def _finalize_results(self, acc):
        """
        Normalizes all accumulated sums and stores them in self.data.
        """
        total_dens = acc['total_density']
        norm = 1.0 / total_dens if total_dens > 1e-12 else 0.0
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
        self._calculate_rdf_and_fortran_comparison(R)

        # Calculate Multipoles (Dipole & Quadrupole)
        #self._calculate_multipoles(X, Y, Z)

        # Calculate 1D & 2D Projections (Averaging)
        self._calculate_projections(X, Y, Z)

    def _generate_coordinates(self):
        """
        Generates 3D Cartesian coordinates (X, Y, Z) and Radius (R).
        RESTORED: Uses the original fast matrix math (tensordot).
        ADJUSTED: Uses float32 to try and fit in RAM.
        """
        nx, ny, nz = self.data.grid_data.shape

        # 1. Use float32 (The only change from the original code)
        # This cuts memory usage by 50% while keeping the fast vectorized math.
        i, j, k = np.indices((nx, ny, nz), dtype=np.float32)

        # Center indices
        di = i - ((nx / 2.0) - 1)
        dj = j - ((ny / 2.0) - 1)
        dk = k - ((nz / 2.0) - 1)

        # Clean up immediately to make room for the big stack
        del i, j, k

        # 2. The "Old" Fast Method (Stack + Tensordot)
        # This is memory heavy but very fast.
        grid_coords = np.stack([di, dj, dk], axis=0)

        # Transform Grid -> Cartesian
        coords = np.tensordot(self.config.transform_matrix, grid_coords, axes=(1, 0))

        # Clean up input immediately
        del grid_coords

        X, Y, Z = coords[0], coords[1], coords[2]
        R = np.sqrt(X ** 2 + Y ** 2 + Z ** 2)

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

    def _calculate_rdf_and_fortran_comparison(self, R):
        """
        Calculates both Density (Legacy) and Probability Mass (Exact/Eq 9) profiles.
        """
        vecs = self.config.lattice_vectors
        box_diagonal_vector = vecs[0] + vecs[1] + vecs[2]
        diagonal = np.linalg.norm(box_diagonal_vector)

        # 1. Determine Bins
        step_vectors = np.linalg.norm(self.config.transform_matrix, axis=0)
        max_voxel_step = np.max(step_vectors)
        nb_bins = int(diagonal / max_voxel_step)
        bins = np.linspace(0, diagonal, nb_bins + 1)

        # 2. Calculate Histograms (WEIGHTED BY DENSITY)
        # This sums the density * voxel_count in each shell.
        # Since sum(density) = 1, this IS the Probability Mass.
        hist_total_mass, bin_edges = np.histogram(R, bins=bins, weights=self.data.grid_data)
        hist_in_vol_mass, _ = np.histogram(R, bins=bins, weights=self.data.density_inside_shape)

        # 3. Calculate Counts (Volume of shell in voxels)
        hist_counts, _ = np.histogram(R, bins=bins)

        self.data.rdf_counts = hist_counts
        self.data.rdf_distance = bin_edges[:-1]
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

        # --- A. LEGACY FORTRAN (Density) ---
        # Density = Mass / Volume. This removes the volume weighting.
        with np.errstate(divide='ignore', invalid='ignore'):
            rho_total = np.nan_to_num(hist_total_mass / hist_counts)
            rho_in_vol = np.nan_to_num(hist_in_vol_mass / hist_counts)

        # Normalize densities to sum to 1 (Legacy behavior)
        norm_rho = np.sum(rho_total)
        if norm_rho > 1e-12:
            self.data.rdf_density_total = rho_total / norm_rho
            self.data.rdf_density_in_vol = rho_in_vol / norm_rho
        else:
            self.data.rdf_density_total = rho_total
            self.data.rdf_density_in_vol = rho_in_vol

        self.data.avg_r_fortran = np.sum(self.data.rdf_density_total * bin_centers)

        # --- B. EXACT PHYSICS (Probability Mass) ---
        # We normalize by the TOTAL mass of the system.
        # This preserves the relative magnitude of "In-Volume" vs "Total".
        total_system_mass = np.sum(hist_total_mass)

        if total_system_mass > 1e-12:
            self.data.rdf_probability_total = hist_total_mass / total_system_mass
            self.data.rdf_probability_in_vol = hist_in_vol_mass / total_system_mass
        else:
            self.data.rdf_probability_total = hist_total_mass
            self.data.rdf_probability_in_vol = hist_in_vol_mass

        # NOTE: Summing self.data.rdf_probability_in_vol NOW gives the integral in Eq 9.

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
        # 1. CT Ratio
        if total_dens > 1e-12:
            self.data.ct_ratio = 1.0 - (acc['masked_density'] / total_dens)
        else:
            self.data.ct_ratio = 0.0

        # 2. Multipoles
        self.data.dipole_moment = acc['dipole'] * norm

        # Symmetrize Quadrupole
        q = acc['quadrupole']
        q[1, 0] = q[0, 1];
        q[2, 0] = q[0, 2];
        q[2, 1] = q[1, 2]
        self.data.quadrupole_moment = q * norm

        # 3. Exact Moments
        self.data.total_weight = total_dens * self.config.dv

        self.data.avg_r_exact = acc['moment_r'] * norm
        self.data.avg_a_exact = acc['moment_a'] * norm
        self.data.avg_b_exact = acc['moment_b'] * norm
        self.data.avg_c_exact = acc['moment_c'] * norm

        self.data.avg_r2_exact = acc['moment_r2'] * norm
        self.data.avg_a2_exact = acc['moment_a2'] * norm
        self.data.avg_b2_exact = acc['moment_b2'] * norm
        self.data.avg_c2_exact = acc['moment_c2'] * norm

        # 4. RDF Finalization
        bins = acc['bins']
        bin_centers = (bins[:-1] + bins[1:]) / 2.0
        self.data.rdf_distance = bins[:-1]
        self.data.rdf_counts = acc['hist_counts']

        hist_counts = acc['hist_counts']
        hist_total = acc['hist_total_mass']
        hist_in = acc['hist_in_vol_mass']

        # A. Legacy (Density)
        with np.errstate(divide='ignore', invalid='ignore'):
            rho_tot = np.nan_to_num(hist_total / hist_counts)
            rho_in = np.nan_to_num(hist_in / hist_counts)

        rho_norm_sum = np.sum(rho_tot)
        if rho_norm_sum > 1e-12:
            self.data.rdf_density_total = rho_tot / rho_norm_sum
            self.data.rdf_density_in_vol = rho_in / rho_norm_sum
        else:
            self.data.rdf_density_total = rho_tot
            self.data.rdf_density_in_vol = rho_in

        self.data.avg_r_fortran = np.sum(self.data.rdf_density_total * bin_centers)

        # B. Exact (Probability Mass)
        mass_norm_sum = np.sum(hist_total)
        if mass_norm_sum > 1e-12:
            self.data.rdf_probability_total = hist_total / mass_norm_sum
            self.data.rdf_probability_in_vol = hist_in / mass_norm_sum
        else:
            self.data.rdf_probability_total = hist_total
            self.data.rdf_probability_in_vol = hist_in