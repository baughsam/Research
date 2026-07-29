from dataclasses import dataclass, field
import numpy as np


@dataclass
class CentralDifference3d:
    dx: float
    dy: float

    vel_x: np.ndarray
    vel_y: np.ndarray

    def upwindDifference(self, array_3d: np.ndarray) -> np.ndarray:
        """
        Calculates the 2nd-Order Central Difference for spatial advection.
        Produces zero numerical diffusion.
        (Method name preserved for RK4 compatibility)
        """
        if array_3d is None or array_3d.ndim != 3:
            raise ValueError("Error: Expected a 3D array.")

        full_derivative_array = np.zeros_like(array_3d)
        Nq = array_3d.shape[2]

        for q in range(Nq):
            vx = self.vel_x[q]
            vy = self.vel_y[q]

            slice_2d = array_3d[:, :, q]
            temp_x = np.zeros_like(slice_2d)
            temp_y = np.zeros_like(slice_2d)

            ### --- 2nd-Order X Differentiation --- ###
            # Central difference for the interior points: (n[i+1] - n[i-1]) / 2dx
            temp_x[1:-1, :] = (slice_2d[2:, :] - slice_2d[:-2, :]) / (2 * self.dx)
            # Fallback to 1st-order at the strict boundaries to prevent index out-of-bounds
            temp_x[0, :] = (slice_2d[1, :] - slice_2d[0, :]) / self.dx
            temp_x[-1, :] = (slice_2d[-1, :] - slice_2d[-2, :]) / self.dx

            x_deriv = -vx * temp_x

            ### --- 2nd-Order Y Differentiation --- ###
            temp_y[:, 1:-1] = (slice_2d[:, 2:] - slice_2d[:, :-2]) / (2 * self.dy)
            temp_y[:, 0] = (slice_2d[:, 1] - slice_2d[:, 0]) / self.dy
            temp_y[:, -1] = (slice_2d[:, -1] - slice_2d[:, -2]) / self.dy

            y_deriv = -vy * temp_y

            # Assemble full derivative
            full_derivative_array[:, :, q] = x_deriv + y_deriv

        return full_derivative_array

#New dataclass to take into account Q space
@dataclass
class UpwindDifference3d:
    dx: float
    dy: float

    vel_x: np.ndarray
    vel_y: np.ndarray

    def upwindDifference(self, array_3d:np.ndarray) -> np.ndarray:
        # Raise Error: Initial array not input
        if array_3d is None:
            raise ValueError("Error: 'array_3d' is empty. Provide a 3D array before calculating.")

        # Raise Error: Initial array incorrect dimensions
        if array_3d.ndim != 3:
            raise ValueError(f"Error: Expected a 3D array, but got a {array_3d.ndim}D array.")

        #Initialize array
        full_derivative_array = np.zeros_like(array_3d)

        # Nq ~ size of 3d dimension in given array
        Nq = array_3d.shape[2]

        # Loop over momentum states. Slice space.
        for q in range(Nq):
            vx = self.vel_x[q]
            vy = self.vel_y[q]

            # Extract 3D spatial slice for specific Q-state
            slice_2d = array_3d[:, :, q]
            temp_x = np.zeros_like(slice_2d)
            temp_y = np.zeros_like(slice_2d)

            ### --- x differentiation --- ###
            if vx > 0:
                temp_x[1:, :] = (slice_2d[1:, :] - slice_2d[:-1, :]) / self.dx
                temp_x[0, :] = (slice_2d[0, :] - 0.0) / self.dx
                x_deriv = -vx * temp_x
            elif vx < 0:
                temp_x[:-1, :] = (slice_2d[1:, :] - slice_2d[:-1, :]) / self.dx
                temp_x[-1, :] = (0.0 - slice_2d[-1, :]) / self.dx
                x_deriv = -vx * temp_x
            else:
                x_deriv = temp_x
            print("x derivative complete.")

            ### --- y differentiation --- ###
            if vy> 0:
                temp_y[:, 1:] = (slice_2d[:, 1:] - slice_2d[:, :-1]) / self.dy
                temp_y[:, 0] = (slice_2d[:, 0] - 0.0) / self.dy
                y_deriv = -vy * temp_y
            elif vy < 0:
                temp_y[:, :-1] = (slice_2d[:, 1:] - slice_2d[:, :-1]) / self.dy
                temp_y[:, -1] = (0.0 - slice_2d[:, -1]) / self.dy
                y_deriv = -vy * temp_y
            else:
                y_deriv = temp_y
            print("y derivative complete.")

            # Put spatial derivative back into 3D tensor
            full_derivative_array[:, :, q] = x_deriv + y_deriv

        return full_derivative_array


@dataclass
class UpwindDifference2d:
    # Spatial Difference
    dx: float
    dy: float

    # Velocities
    velocity_x: float = 0.0
    velocity_y: float = 0.0


    def upwindDifference(self, array_2d: np.ndarray) -> np.ndarray:
        # Raise Error: Initial array not input
        if array_2d is None:
            raise ValueError("Error: 'array_2d' is empty. Provide a 2D array before calculating.")

        # Raise Error: Initial array incorrect dimensions
        if array_2d.ndim != 2:
            raise ValueError(f"Error: Expected a 2D array, but got a {array_2d.ndim}D array.")

        # Initializing zero array for slice derivation
        print("Initializing temp arrays.")
        temp_x = np.zeros_like(array_2d)
        temp_y = np.zeros_like(array_2d)

        ### --- x differentiation --- ###
        if self.velocity_x > 0:
            temp_x[1:, :] = (array_2d[1:, :] - array_2d[:-1, :]) / self.dx
            temp_x[0, :] = (array_2d[0, :] - 0.0) / self.dx
            x_deriv_array = -self.velocity_x * temp_x
        elif self.velocity_x < 0:
            temp_x[:-1, :] = (array_2d[1:, :] - array_2d[:-1, :]) / self.dx
            temp_x[-1, :] = (0.0 - array_2d[-1, :]) / self.dx
            x_deriv_array = -self.velocity_x * temp_x
        else:
            x_deriv_array = temp_x
        print("x derivative complete.")

        ### --- y differentiation --- ###
        if self.velocity_y > 0:
            temp_y[:, 1:] = (array_2d[:, 1:] - array_2d[:, :-1]) / self.dy
            temp_y[:, 0] = (array_2d[:, 0] - 0.0) / self.dy
            y_deriv_array = -self.velocity_y * temp_y
        elif self.velocity_y < 0:
            temp_y[:, :-1] = (array_2d[:, 1:] - array_2d[:, :-1]) / self.dy
            temp_y[:, -1] = (0.0 - array_2d[:, -1]) / self.dy
            y_deriv_array = -self.velocity_y * temp_y
        else:
            y_deriv_array = temp_y
        print("y derivative complete.")

        full_derivative_array = x_deriv_array + y_deriv_array
        print("Upwind Difference Complete.")
        print("Returning differentiated array.")

        return full_derivative_array

@dataclass
class RungeKutta4:
    # User Inputs
    spatial_solver: object            # UpwindDifference2d instance
    total_sim_time: float
    courant_number: float = 0.5       # Default Safety factor for stability
    scattering_solver: object = None  # scattering objects in k_scat.py

    # Calculated Attributes
    dt: float = field(init=False)
    num_steps: int = field(init=False)

    def __post_init__(self):
        """
        Auto-calculates safe time step
        """
        # Extract physics parameters from Upwind object
        dx = self.spatial_solver.dx
        dy = self.spatial_solver.dy

        if hasattr(self.spatial_solver, 'vel_x'):
            vx_max = np.max(np.abs(self.spatial_solver.vel_x))
            vy_max = np.max(np.abs(self.spatial_solver.vel_y))
        else:
            vx_max = abs(self.spatial_solver.velocity_x)
            vy_max = abs(self.spatial_solver.velocity_y)


        # Calculates max safe time step using CFL Condition
        self.dt = self.courant_number / ((vx_max/dx) + (vy_max/dy) + 1e-15)
        # Calculate how many total loop iterations are needed for the simulation
        self.num_steps = int(self.total_sim_time / self.dt)
        print(f"Calculated dt: {self.dt:.4f} fs | Total Steps: {self.num_steps}")

    def solve(self, n_initial: np.ndarray, save_interval: int = 10) -> np.ndarray:
        """
        Executes the explicit RK4 time integration.
        """
        n_current = np.copy(n_initial)
        history = [np.copy(n_current)]

        for step in range(self.num_steps):

            k1_scat = self.scattering_solver.calc_scattering(n_current) if self.scattering_solver else 0.0
            k1 = self.spatial_solver.upwindDifference(n_current) + k1_scat


            n_temp1 = n_current + k1 * (self.dt / 2.0)
            k2_scat = self.scattering_solver.calc_scattering(n_temp1) if self.scattering_solver else 0.0
            k2 = self.spatial_solver.upwindDifference(n_temp1) + k2_scat

            n_temp2 = n_current + k2 * (self.dt / 2.0)
            k3_scat = self.scattering_solver.calc_scattering(n_temp2) if self.scattering_solver else 0.0
            k3 = self.spatial_solver.upwindDifference(n_temp2) + k3_scat

            n_temp3 = n_current + k3 * self.dt
            k4_scat = self.scattering_solver.calc_scattering(n_temp3) if self.scattering_solver else 0.0
            k4 = self.spatial_solver.upwindDifference(n_temp3) + k4_scat

            n_current = n_current + (self.dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

            # Save frames for the animation
            if (step + 1) % save_interval == 0:
                history.append(np.copy(n_current))

        return history

    def solve_hdf5_compressed(self, n_initial: np.ndarray, save_interval: int = 10, output_file: str = "full_history.h5") -> str:
        """
        Executes the explicit RK4 time integration.
        Streams the full 3D tensor directly to an HDF5 file on disk to prevent RAM overflow.
        """
        import h5py

        n_current = np.copy(n_initial)
        Nx, Ny, Nq = n_current.shape

        print(f"Streaming full 3D tensor history to {output_file} to preserve RAM...")

        with h5py.File(output_file, 'w') as f:
            # Create a dynamically resizable dataset chunked for performance
            dset = f.create_dataset(
                "frames",
                shape=(1, Nx, Ny, Nq),
                maxshape=(None, Nx, Ny, Nq),
                dtype=np.float32,
                compression="gzip",  # Shrinks the file size on disk
                compression_opts=4
            )
            f.attrs['dt'] = self.dt

            # Save the initial conditions
            dset[0] = n_current.astype(np.float32)
            frame_count = 1

            for step in range(self.num_steps):

                k1_scat = self.scattering_solver.calc_scattering(n_current) if self.scattering_solver else 0.0
                k1 = self.spatial_solver.upwindDifference(n_current) + k1_scat

                n_temp1 = n_current + k1 * (self.dt / 2.0)
                k2_scat = self.scattering_solver.calc_scattering(n_temp1) if self.scattering_solver else 0.0
                k2 = self.spatial_solver.upwindDifference(n_temp1) + k2_scat

                n_temp2 = n_current + k2 * (self.dt / 2.0)
                k3_scat = self.scattering_solver.calc_scattering(n_temp2) if self.scattering_solver else 0.0
                k3 = self.spatial_solver.upwindDifference(n_temp2) + k3_scat

                n_temp3 = n_current + k3 * self.dt
                k4_scat = self.scattering_solver.calc_scattering(n_temp3) if self.scattering_solver else 0.0
                k4 = self.spatial_solver.upwindDifference(n_temp3) + k4_scat

                n_current = n_current + (self.dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

                # Stream to disk and discard from RAM
                if (step + 1) % save_interval == 0:
                    dset.resize(frame_count + 1, axis=0)  # Expand the file by 1 frame
                    dset[frame_count] = n_current.astype(np.float32)  # Write directly to disk
                    frame_count += 1

        print("Integration complete. Full phase-space history safely secured on disk.")
        return output_file  # Return the file path instead of the array list
