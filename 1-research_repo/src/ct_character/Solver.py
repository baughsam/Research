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
            # This saves significant CPU time when just calculating CT Ratio.
            if self.do_rdf:
                X, Y, Z, R = self._generate_slice_coords(k, nz, xy_base_coords, z_step_vec)
            else:
                X, Y, Z, _ = self._generate_slice_coords(k, nz, xy_base_coords, z_step_vec, calc_r=False)

            # C. Create Mask for this slice
            mask_slice = self._create_volume_mask(X, Y, Z)

            # D. Apply Mask (Isolate density inside the shape)
            rho_inside = np.where(mask_slice, rho_slice, 0.0)

            # --- ACCUMULATION PHASE ---

            # 1. CT Ratio (Always Run)
            self._accumulate_ct_ratio(acc, rho_slice, rho_inside)

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
            # 1. Counts (Needed for Legacy "Avg Density" calculation)
            acc['hist_counts'] = np.zeros(nb_bins)
            # 2. Mass (Needed for Correct "Probability" calculation)
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

    # --- ACCUMULATION METHODS ---

    def _accumulate_ct_ratio(self, acc, rho_slice, rho_inside):
        """Updates raw density running totals."""
        acc['total_density'] += np.sum(rho_slice)
        acc['masked_density'] += np.sum(rho_inside)

    def _accumulate_rdf(self, acc, R, rho_slice, rho_inside):
        """
        Updates RDF histograms.
        Tracks BOTH counts (for Legacy) and mass (for Physics).
        """
        bins = acc['bins']

        # 1. Histogram Voxel Counts (Legacy: "Avg Density per Voxel")
        counts, _ = np.histogram(R, bins=bins)

        # 2. Histogram Density Mass (Physics: "Probability Mass")
        mass_tot, _ = np.histogram(R, bins=bins, weights=rho_slice)
        mass_in, _ = np.histogram(R, bins=bins, weights=rho_inside)

        acc['hist_counts'] += counts
        acc['hist_total_mass'] += mass_tot
        acc['hist_in_vol_mass'] += mass_in

    def _finalize_results(self, acc):
        dv = self.config.dv

        #  CT Ratio
        total_electrons = acc['total_density'] * dv
        masked_electrons = acc['masked_density'] * dv

        print(f"  > Integration Check:")
        print(f"    Total Charge: {total_electrons:.4f}")
        print(f"    Mask Charge:  {masked_electrons:.4f}")

        if total_electrons > 1e-12:
            self.data.ct_ratio = 1.0 - (masked_electrons / total_electrons)
        else:
            self.data.ct_ratio = 0.0

        self.data.total_weight = total_electrons

        # RDF Finalization
        if self.do_rdf:
            bins = acc['bins']
            self.data.rdf_distance = bins[:-1]
            self.data.rdf_counts = acc['hist_counts']

            hist_counts = acc['hist_counts']
            hist_total = acc['hist_total_mass']
            hist_in = acc['hist_in_vol_mass']

            # Legacy (Fortran) Density
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

            # Correct Probability
            mass_norm_sum = acc['total_density']
            if mass_norm_sum > 1e-12:
                self.data.rdf_probability_total = hist_total / mass_norm_sum
                self.data.rdf_probability_in_vol = hist_in / mass_norm_sum
            else:
                self.data.rdf_probability_total = hist_total
                self.data.rdf_probability_in_vol = hist_in