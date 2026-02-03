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
            # We pass 'acc' to these methods so they can update the running totals
            self._accumulate_ct_ratio(acc, rho_slice, rho_inside)
            self._accumulate_rdf(acc, R, rho_slice, rho_inside)
            self._accumulate_multipoles(acc, X, Y, Z, R, rho_inside)
            self._accumulate_projections(acc, X, Y, Z, R, rho_inside)

        # 4. Finalize (Normalize sums and store in self.data)
        self._finalize_results(acc)
        print("Solver finished successfully.")

    def build_visual_mask(self):
        """
        NEW METHOD: Reconstructs the 3D boolean mask for visualization
        slice-by-slice to avoid RAM spikes.
        """
        print("  > Re-generating mask for visualization (Slice-by-Slice)...")
        nx, ny, nz = self.data.grid_data.shape
        full_mask = np.zeros((nx, ny, nz), dtype=bool)

        xy_base = self._precompute_xy_grid(nx, ny)
        z_step_vec = self.config.transform_matrix[:, 2]

        for k in range(nz):
            X, Y, Z, _ = self._generate_slice_coords(k, nz, xy_base, z_step_vec)
            full_mask[:, :, k] = self._create_volume_mask(X, Y, Z)

        return full_mask

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