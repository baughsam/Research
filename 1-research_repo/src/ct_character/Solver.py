import numpy as np


class Solver:
    def __init__(self, exciton_data, configuration, do_rdf_analysis=False):
        """
        Initializes the Solver.
        :param do_rdf_analysis: If False, skips binning and R-calculation for speed.
        """
        self.data = exciton_data
        self.config = configuration
        self.do_rdf = do_rdf_analysis

    def solve(self):
        """
        Orchestrates the analysis using Memory-Optimized Z-Plane Slicing.
        """
        mode_str = "ON" if self.do_rdf else "OFF"
        print(f"Running Solver (In-Volume Analysis: {mode_str})...")

        nx, ny, nz = self.data.grid_data.shape

        # 1. Initialize Accumulators
        acc = self._initialize_accumulators()

        # 2. Pre-calculate the XY Grid (Constant for all Z-planes)
        xy_base_coords = self._precompute_xy_grid(nx, ny)
        z_step_vec = self.config.transform_matrix[:, 2]

        # 3. Z-Plane Loop (The Memory Saver)
        for k in range(nz):
            # A. Get Data for this slice
            rho_slice = self.data.grid_data[:, :, k]

            # B. Generate Coordinates
            # OPTIMIZATION: Only calculate 'R' (Radius) if we are doing binning.
            if self.do_rdf:
                X, Y, Z, R = self._generate_slice_coords(k, nz, xy_base_coords, z_step_vec)
            else:
                X, Y, Z, _ = self._generate_slice_coords(k, nz, xy_base_coords, z_step_vec, calc_r=False)

            # C. Create Mask for this slice
            mask_slice = self._create_volume_mask(X, Y, Z)

            # D. Apply Mask (Isolate density inside the shape)
            rho_inside = np.where(mask_slice, rho_slice, 0.0)

            # --- ACCUMULATION PHASE ---
            # 1. Raw Density sums (Always Run)
            self._accumulate_raw_sums(acc, rho_slice, rho_inside)

            # 2. In-Volume Analysis (Conditional)
            if self.do_rdf:
                self._accumulate_rdf(acc, R, rho_slice, rho_inside)

        # 4. Finalize (Normalize sums and store in self.data)
        self._finalize_results(acc)
        print("Solver finished successfully.")

    def build_visual_mask(self):
        """
        Reconstructs the 3D boolean mask for visualization slice-by-slice.
        """
        print("  > Re-generating mask for visualization (Slice-by-Slice)...")
        nx, ny, nz = self.data.grid_data.shape
        full_mask = np.zeros((nx, ny, nz), dtype=bool)

        xy_base = self._precompute_xy_grid(nx, ny)
        z_step_vec = self.config.transform_matrix[:, 2]

        for k in range(nz):
            X, Y, Z, _ = self._generate_slice_coords(k, nz, xy_base, z_step_vec, calc_r=False)
            full_mask[:, :, k] = self._create_volume_mask(X, Y, Z)

        return full_mask

    def _initialize_accumulators(self):
        """Prepares zeroed variables to hold sums."""
        acc = {
            'total_density': 0.0,
            'masked_density': 0.0
        }

        # Only set up histogram bins if we are doing In-Volume analysis
        if self.do_rdf:
            vecs = self.config.lattice_vectors
            box_diag = np.linalg.norm(vecs[0] + vecs[1] + vecs[2])
            step_vecs = np.linalg.norm(self.config.transform_matrix, axis=0)
            max_step = np.max(step_vecs)

            # Create bins
            nb_bins = int(box_diag / max_step)
            acc['bins'] = np.linspace(0, box_diag, nb_bins + 1)

            # Arrays for Accumulation
            acc['hist_counts'] = np.zeros(nb_bins)
            acc['hist_total_mass'] = np.zeros(nb_bins)
            acc['hist_in_vol_mass'] = np.zeros(nb_bins)

        return acc

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

    def _generate_slice_coords(self, k, nz, xy_base, z_step_vec, calc_r=True):
        """Adds the Z-component to the precomputed XY grid for the current slice."""
        dk = float(k) - ((nz / 2.0) - 1)
        current_coords = xy_base + z_step_vec[:, np.newaxis, np.newaxis] * dk

        X = current_coords[0]
        Y = current_coords[1]
        Z = current_coords[2]

        R = None
        if calc_r:
            R = np.sqrt(X ** 2 + Y ** 2 + Z ** 2)

        return X, Y, Z, R

    def _create_volume_mask(self, X, Y, Z):
        """Asks the Shape object to create boolean mask for this slice."""
        return self.config.shape.is_inside(X, Y, Z)

    def _accumulate_raw_sums(self, acc, rho_slice, rho_inside):
        """Updates raw density running totals."""
        acc['total_density'] += np.sum(rho_slice)
        acc['masked_density'] += np.sum(rho_inside)

    def _accumulate_rdf(self, acc, R, rho_slice, rho_inside):
        """Updates RDF histograms."""
        bins = acc['bins']

        # 1. Histogram Voxel Counts
        counts, _ = np.histogram(R, bins=bins)

        # 2. Histogram Density Mass
        mass_tot, _ = np.histogram(R, bins=bins, weights=rho_slice)
        mass_in, _ = np.histogram(R, bins=bins, weights=rho_inside)

        acc['hist_counts'] += counts
        acc['hist_total_mass'] += mass_tot
        acc['hist_in_vol_mass'] += mass_in

    def _finalize_results(self, acc):
        dv = self.config.dv

        # Legacy total weight (often outputs ~0.03 for probabilities due to *dv)
        # Keeping this intact so the _OUT file doesn't break backwards compatibility
        total_electrons = acc['total_density'] * dv
        masked_electrons = acc['masked_density'] * dv

        print(f"  > Integration Check:")
        print(f"    Raw Mass (Total): {acc['total_density']:.4f}")
        print(f"    Raw Mass (Mask):  {acc['masked_density']:.4f}")

        self.data.total_weight = total_electrons

        # RDF Finalization and NEW CT RATIO MATH
        if self.do_rdf:
            bins = acc['bins']
            self.data.rdf_distance = bins[:-1]
            self.data.rdf_counts = acc['hist_counts']

            hist_counts = acc['hist_counts']
            hist_total = acc['hist_total_mass']
            hist_in = acc['hist_in_vol_mass']

            # Correct Probability (Shell Probabilities)
            mass_norm_sum = acc['total_density']
            if mass_norm_sum > 1e-12:
                self.data.rdf_probability_total = hist_total / mass_norm_sum
                self.data.rdf_probability_in_vol = hist_in / mass_norm_sum
            else:
                self.data.rdf_probability_total = hist_total
                self.data.rdf_probability_in_vol = hist_in

            # -----------------------------------------------------------------
            # NEW: Calculate Volume-Corrected CT Ratio
            # This directly calculates the plateau of the Volume-Corrected graph
            # -----------------------------------------------------------------
            valid = hist_counts > 0
            dens_in_shell = np.zeros_like(self.data.rdf_probability_in_vol)
            dens_tot_shell = np.zeros_like(self.data.rdf_probability_total)

            dens_in_shell[valid] = self.data.rdf_probability_in_vol[valid] / hist_counts[valid]
            dens_tot_shell[valid] = self.data.rdf_probability_total[valid] / hist_counts[valid]

            sum_in = np.sum(dens_in_shell)
            sum_tot = np.sum(dens_tot_shell)

            if sum_tot > 1e-12:
                # The plateau ratio!
                self.data.ct_ratio = sum_in / sum_tot
            else:
                self.data.ct_ratio = 0.0

            # Legacy (Fortran) Density fallback
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

            # Calculate Standard CDF
            self.data.cdf_total = np.cumsum(self.data.rdf_probability_total)
            self.data.cdf_in_vol = np.cumsum(self.data.rdf_probability_in_vol)

            print(f"  > NEW Volume-Corrected CT Ratio: {self.data.ct_ratio:.6f}")

        else:
            # Fallback if RDF is not calculated: Pure 3D Volumetric Ratio inside shape
            if total_electrons > 1e-12:
                self.data.ct_ratio = masked_electrons / total_electrons
            else:
                self.data.ct_ratio = 0.0