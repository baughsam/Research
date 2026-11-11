import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter
import functools


class _BandSummedFrames:
    """Lazy view over multi-band frames (n_bands, Nx, Ny, Nq) that yields band-summed
    (Nx, Ny, Nq) frames, so legacy single-band routines work unchanged on 4D data.
    Works with in-memory lists/arrays and h5py datasets (one frame read at a time)."""

    def __init__(self, frames):
        self._frames = frames

    def __len__(self):
        return len(self._frames)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return [self[i] for i in range(*idx.indices(len(self)))]
        return np.sum(np.asarray(self._frames[idx]), axis=0)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]


def _is_multiband(frames):
    return np.ndim(frames[0]) == 4 if not hasattr(frames[0], 'ndim') else frames[0].ndim == 4


def _accepts_multiband(func):
    """Decorator: if `frames` holds 4D (n_bands, Nx, Ny, Nq) frames, sum over bands first."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if args:
            if _is_multiband(args[0]):
                args = (_BandSummedFrames(args[0]),) + args[1:]
        elif 'frames' in kwargs and _is_multiband(kwargs['frames']):
            kwargs['frames'] = _BandSummedFrames(kwargs['frames'])
        return func(*args, **kwargs)
    return wrapper



@_accepts_multiband
def export_diffusion_gif_updated2(frames, dt, save_interval, length_x, length_y, grid_x, grid_y,
                                 right_panel_mode='qgrid', target_state=None,
                                 q_vectors=None, projection=('x', 'y'),
                                 filename="diffusion.gif"):
    """
    Exports a two-panel animated GIF.
    Optimized to eliminate flickering by finding true global limits,
    and uses pre-calculated projections to speed up rendering.
    Also capping the physical limits in the visualization gives a more
    intuitive picture of "exciton mass" flow across momentum and spatial grids.
    """
    print(f"Rendering two-panel GIF to {filename} (This might take a minute)...")

    Nq = frames[0].shape[2]

    if right_panel_mode == 'slice':
        if target_state is None or target_state < 0 or target_state >= Nq:
            raise ValueError(f"Invalid target_state. Must be between 0 and {Nq - 1}.")

    elif right_panel_mode == 'qgrid':
        if q_vectors is None:
            raise ValueError("q_vectors must be provided for 3D qgrid projection.")

        axis_map = {'x': 0, 'y': 1, 'z': 2}
        ax1_idx = axis_map[projection[0]]
        ax2_idx = axis_map[projection[1]]

        unique_q1 = np.unique(q_vectors[:, ax1_idx])
        unique_q2 = np.unique(q_vectors[:, ax2_idx])

        dq1 = (unique_q1[1] - unique_q1[0]) if len(unique_q1) > 1 else 1.0
        dq2 = (unique_q2[1] - unique_q2[0]) if len(unique_q2) > 1 else 1.0
        q_extent = [unique_q1.min() - dq1 / 2, unique_q1.max() + dq1 / 2,
                    unique_q2.min() - dq2 / 2, unique_q2.max() + dq2 / 2]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    physical_time_per_frame = dt * save_interval

    # --- 1. SETUP LEFT PANEL (REAL SPACE) ---
    print("Locking Real-Space scale...")
    # The spatial edges drop to true zero, so vmin=0 is physically correct here
    global_max_spatial = np.max(np.sum(frames[0], axis=2))

    init_total_density = np.sum(frames[0], axis=2)
    im1 = ax1.imshow(init_total_density.T, origin='lower', cmap='magma',
                     extent=[0, length_x, 0, length_y], interpolation='nearest',
                     vmin=0, vmax=global_max_spatial)

    ax1.set_xlabel("X Position (nm)")
    ax1.set_ylabel("Y Position (nm)")
    fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04).set_label("Total Mass")

    # --- 2. SETUP RIGHT PANEL & PRE-CALCULATE ---
    print("Pre-calculating Q-Grid history to lock true contrast limits...")
    if right_panel_mode == 'slice':
        # Fast extraction of the target slice across all frames
        slice_history = [f[:, :, target_state] for f in frames]
        vmin_slice = np.min(slice_history)
        vmax_slice = np.max(slice_history)

        im2 = ax2.imshow(slice_history[0].T, origin='lower', cmap='viridis',
                         extent=[0, length_x, 0, length_y], interpolation='nearest',
                         vmin=vmin_slice, vmax=vmax_slice)

        ax2.set_xlabel("X Position (nm)")
        ax2.set_ylabel("Y Position (nm)")
        fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04).set_label(f"Q-State {target_state} Mass")

    elif right_panel_mode == 'qgrid':
        q_history = []
        for f in frames:
            # Sum spatial axes (0 and 1) to get total mass per Q-state[cite: 7]
            mass_per_state = np.sum(f, axis=(0, 1))

            momentum_2d = np.zeros((len(unique_q1), len(unique_q2)))
            for q_idx in range(Nq):
                val1 = q_vectors[q_idx, ax1_idx]
                val2 = q_vectors[q_idx, ax2_idx]
                idx1 = np.where(unique_q1 == val1)[0][0]
                idx2 = np.where(unique_q2 == val2)[0][0]
                momentum_2d[idx1, idx2] += mass_per_state[q_idx]

            q_history.append(momentum_2d)

        # Lock to the absolute minimum and maximum found in the simulation
        vmin_mom = np.min(q_history)
        vmax_mom = np.max(q_history)

        im2 = ax2.imshow(q_history[0].T, origin='lower', cmap='plasma',
                         extent=q_extent, interpolation='nearest',
                         vmin=vmin_mom, vmax=vmax_mom)

        ax2.set_xlabel(f"q_{projection[0]}")
        ax2.set_ylabel(f"q_{projection[1]}")
        fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04).set_label("Projected Mass in Q-Plane")

    # --- 3. FAST UPDATE FUNCTION ---
    def update(frame_idx):
        current_time_fs = frame_idx * physical_time_per_frame

        # Update left panel
        total_density = np.sum(frames[frame_idx], axis=2)
        im1.set_data(total_density.T)

        current_max = np.max(total_density)
        im1.set_clim(vmin=0, vmax=current_max)

        ax1.set_title(f"Total Exciton Density | Time: {current_time_fs:.3f} fs")

        # Instantly update right panel using the pre-calculated history
        if right_panel_mode == 'slice':
            im2.set_data(slice_history[frame_idx].T)
            ax2.set_title(f"Phase Space Slice [State {target_state}]")

        elif right_panel_mode == 'qgrid':
            im2.set_data(q_history[frame_idx].T)
            ax2.set_title(f"Momentum Space Projection ({projection[0]}-{projection[1]} Plane)")

        return [im1, im2]

    # Render animation
    ani = animation.FuncAnimation(fig, update, frames=len(frames), blit=False)
    ani.save(filename, writer=PillowWriter(fps=20))

    plt.close(fig)
    print(f"GIF saved successfully as '{filename}'!")



@_accepts_multiband
def export_diffusion_gif_updated1(frames, dt, save_interval, length_x, length_y, grid_x, grid_y,
                         right_panel_mode='qgrid', target_state=None,
                         q_vectors=None, projection=('x', 'y'),  # <-- New Arguments
                         filename="diffusion.gif"):
    """
    Exports a two-panel animated GIF of the spatial density and momentum distribution.
    Supports 3D momentum grids via projection.
    """
    print(f"Rendering two-panel GIF to {filename} (This might take a minute)...")

    Nq = frames[0].shape[2]

    if right_panel_mode == 'slice':
        if target_state is None or target_state < 0 or target_state >= Nq:
            raise ValueError(f"Invalid target_state. Must be between 0 and {Nq - 1}.")

    elif right_panel_mode == 'qgrid':
        if q_vectors is None:
            raise ValueError("q_vectors must be provided for 3D qgrid projection.")

        # Setup projection axes
        axis_map = {'x': 0, 'y': 1, 'z': 2}
        ax1_idx = axis_map[projection[0]]
        ax2_idx = axis_map[projection[1]]

        unique_q1 = np.unique(q_vectors[:, ax1_idx])
        unique_q2 = np.unique(q_vectors[:, ax2_idx])

        dq1 = (unique_q1[1] - unique_q1[0]) if len(unique_q1) > 1 else 1.0
        dq2 = (unique_q2[1] - unique_q2[0]) if len(unique_q2) > 1 else 1.0
        q_extent = [unique_q1.min() - dq1 / 2, unique_q1.max() + dq1 / 2,
                    unique_q2.min() - dq2 / 2, unique_q2.max() + dq2 / 2]

    # Recreate Spatial Grids
    delta_x = length_x / (grid_x - 1)
    delta_y = length_y / (grid_y - 1)

    X_grid = np.empty((grid_x, grid_y))
    Y_grid = np.empty((grid_x, grid_y))
    for i in range(grid_x):
        for j in range(grid_y):
            X_grid[i, j] = i * delta_x
            Y_grid[i, j] = j * delta_y

    X_flat = X_grid.flatten()
    Y_flat = Y_grid.flatten()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    physical_time_per_frame = dt * save_interval
    colorbar_created = False

    def update(frame_idx):
        nonlocal colorbar_created
        ax1.clear()
        ax2.clear()

        current_time_fs = frame_idx * physical_time_per_frame
        frame = frames[frame_idx]

        # --- LEFT PANEL: Real Space ---
        total_density = np.sum(frame, axis=2)
        sc1 = ax1.scatter(X_flat, Y_flat, c=total_density.flatten(), cmap='magma', marker='s', s=15)

        ax1.set_title(f"Total Exciton Density | Time: {current_time_fs:.3f} fs")
        ax1.set_xlabel("X Position (nm)")
        ax1.set_ylabel("Y Position (nm)")
        ax1.set_xlim(0, length_x)
        ax1.set_ylim(0, length_y)

        # --- RIGHT PANEL: Momentum Space ---
        if right_panel_mode == 'slice':
            slice_density = frame[:, :, target_state]
            sc2 = ax2.scatter(X_flat, Y_flat, c=slice_density.flatten(), cmap='viridis', marker='s', s=15)
            ax2.set_title(f"Phase Space Slice [State {target_state}]")
            ax2.set_xlabel("X Position (nm)")
            ax2.set_ylabel("Y Position (nm)")
            ax2.set_xlim(0, length_x)
            ax2.set_ylim(0, length_y)

        elif right_panel_mode == 'qgrid':
            momentum_2d = np.zeros((len(unique_q1), len(unique_q2)))

            # Sum out the unobserved axis to project 3D down to 2D
            for q_idx in range(Nq):
                density_in_state = np.sum(frame[:, :, q_idx])
                val1 = q_vectors[q_idx, ax1_idx]
                val2 = q_vectors[q_idx, ax2_idx]
                idx1 = np.where(unique_q1 == val1)[0][0]
                idx2 = np.where(unique_q2 == val2)[0][0]
                momentum_2d[idx1, idx2] += density_in_state

            sc2 = ax2.imshow(momentum_2d.T, origin='lower', cmap='plasma',
                             extent=q_extent, interpolation='nearest')

            ax2.set_title(f"Momentum Space Projection ({projection[0]}-{projection[1]} Plane)")
            ax2.set_xlabel(f"q_{projection[0]}")
            ax2.set_ylabel(f"q_{projection[1]}")

        # Create colorbars
        if not colorbar_created:
            fig.colorbar(sc1, ax=ax1, fraction=0.046, pad=0.04).set_label("Total Mass")
            fig.colorbar(sc2, ax=ax2, fraction=0.046, pad=0.04).set_label(
                "Projected Mass in Q-Plane" if right_panel_mode == 'qgrid' else "Q-State Mass")
            colorbar_created = True

    # Compile and Save
    ani = animation.FuncAnimation(fig, update, frames=len(frames), blit=False)
    ani.save(filename, writer=PillowWriter(fps=20))

    plt.close(fig)
    print(f"GIF saved successfully as '{filename}'!")

@_accepts_multiband
def export_diffusion_gif(frames, dt, save_interval, length_x, length_y, grid_x, grid_y,
                         right_panel_mode='qgrid', target_state=None, q_Nx=None, q_Ny=None,
                         filename="diffusion.gif"):
    """
    Exports a two-panel animated GIF of the spatial density and momentum distribution.
    """
    print(f"Rendering two-panel GIF to {filename} (This might take a minute)...")

    # 1. Validation / Setup for Q-Grid
    Nq = frames[0].shape[2]
    if right_panel_mode == 'slice':
        if target_state is None or target_state < 0 or target_state >= Nq:
            raise ValueError(f"Invalid target_state. Must be between 0 and {Nq - 1}.")
    elif right_panel_mode == 'qgrid':
        if q_Nx is None or q_Ny is None:
            q_Nx = int(np.sqrt(Nq))
            q_Ny = Nq // q_Nx

    # 2. Recreate Spatial Grids
    delta_x = length_x / (grid_x - 1)
    delta_y = length_y / (grid_y - 1)

    X_grid = np.empty((grid_x, grid_y))
    Y_grid = np.empty((grid_x, grid_y))
    for i in range(grid_x):
        for j in range(grid_y):
            X_grid[i, j] = i * delta_x
            Y_grid[i, j] = j * delta_y

    X_flat = X_grid.flatten()
    Y_flat = Y_grid.flatten()

    # 3. Setup the Two-Panel Figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    physical_time_per_frame = dt * save_interval

    # Draw initial colorbars outside the loop so the plot doesn't shrink during animation
    colorbar_created = False

    # 4. Define the Animation Update Function
    def update(frame_idx):
        nonlocal colorbar_created
        ax1.clear()
        ax2.clear()

        current_time_fs = frame_idx * physical_time_per_frame
        frame = frames[frame_idx]

        # --- LEFT PANEL: Real Space ---
        total_density = np.sum(frame, axis=2)
        sc1 = ax1.scatter(X_flat, Y_flat, c=total_density.flatten(), cmap='magma', marker='s', s=15)

        ax1.set_title(f"Total Exciton Density | Time: {current_time_fs:.3f} fs")
        ax1.set_xlabel("X Position (nm)")
        ax1.set_ylabel("Y Position (nm)")
        ax1.set_xlim(0, length_x)
        ax1.set_ylim(0, length_y)

        # --- RIGHT PANEL: Momentum Space ---
        if right_panel_mode == 'slice':
            slice_density = frame[:, :, target_state]
            sc2 = ax2.scatter(X_flat, Y_flat, c=slice_density.flatten(), cmap='viridis', marker='s', s=15)
            ax2.set_title(f"Phase Space Slice [State {target_state}]")
            ax2.set_xlabel("X Position (nm)")
            ax2.set_ylabel("Y Position (nm)")
            ax2.set_xlim(0, length_x)
            ax2.set_ylim(0, length_y)

        elif right_panel_mode == 'qgrid':
            momentum_dist = np.sum(frame, axis=(0, 1)).reshape((q_Nx, q_Ny))
            sc2 = ax2.imshow(momentum_dist.T, origin='lower', cmap='plasma',
                             extent=[-np.pi, np.pi, -np.pi, np.pi], interpolation='nearest')
            ax2.set_title("Momentum Space (Q-Grid)")
            ax2.set_xlabel("qx")
            ax2.set_ylabel("qy")

        # Create colorbars only on the very first frame
        if not colorbar_created:
            fig.colorbar(sc1, ax=ax1, fraction=0.046, pad=0.04).set_label("Total Mass")
            fig.colorbar(sc2, ax=ax2, fraction=0.046, pad=0.04).set_label("Q-State Mass")
            colorbar_created = True

    # 5. Compile and Save
    ani = animation.FuncAnimation(fig, update, frames=len(frames), blit=False)
    ani.save(filename, writer=PillowWriter(fps=20))

    plt.close(fig)
    print(f"GIF saved successfully as '{filename}'!")

@_accepts_multiband
def extract_diffusion_constant(frames: list, dt: float, save_interval: int,
                               x_grid: np.ndarray, y_grid: np.ndarray,
                               cutoff_fraction: float = 0.5,
                               show_plot: bool = True):
    """
    Extracts macrocopic diffusion constant (D) from the history of spatial density frames
    :param frames: List of 3D numpy arrays (the history of the simulation)
    :param dt: The time step of simulation (femtoseconds)
    :param save_interval: How often frames are being saved
    :param x_grid: 2D array of x coords.
    :param y_grid: 2D array of y coords.
    :param cutoff_fraction: the fraction of the simulation to skip to ensure we are in the diffusive regime
    :param shot_plot: Boolean ot display the verification plot
    :return:  Diffusion constant D (nm^2/fs)
    """

    print ("Extracting macroscopic diffusion constant...")

    times_fs = []
    msd_nm2 = []
    physical_time_per_frame = dt * save_interval

    # Calculate MSD for each frame
    for i, frame in enumerate(frames):
        current_time_fs = i * physical_time_per_frame

        # Get physical 2D density
        total_density = np.sum(frame, axis=2)
        total_mass = np.sum(total_density)

        if total_mass == 0:
            continue

        # Center of mass
        x_cm = np.sum(total_density * x_grid) / total_mass
        y_cm = np.sum(total_density * y_grid) / total_mass

        # Spatial Variance (MSD)
        r_squared = (x_grid -x_cm)**2 + (y_grid -y_cm)**2
        current_msd = np.sum(total_density * r_squared) / total_mass

        times_fs.append(current_time_fs)
        msd_nm2.append(current_msd)

    # List to Arrays
    times_fs=np.array(times_fs)
    msd_nm2 = np.array(msd_nm2)

    # Linear Regression on the Diffusive Regime
    cutoff_index = int(len(times_fs) * cutoff_fraction)
    fit_times = times_fs[cutoff_index:]
    fit_msd = msd_nm2[cutoff_index:]

    #Troubleshooting
    print(f"Number of points to fit: {len(fit_times)}")
    print(f"Contains NaNs: {np.any(np.isnan(fit_msd))}")
    print(f"Contains Infs: {np.any(np.isinf(fit_msd))}")

    # Fit a 1st-degree polynomial: MSD = (4D) * t + intercept
    slope, intercept = np.polyfit(fit_times, fit_msd, 1)
    D_extracted = slope / 4.0

    print(f"Extraction Complete. D = {D_extracted:.4e} nm^2/fs")

    # Verification Plot
    if show_plot:
        plt.figure(figsize=(8, 6))
        plt.plot(times_fs, msd_nm2, 'b-', linewidth=2, label="Simulated MSD")
        plt.plot(fit_times, (slope * fit_times) + intercept, 'r--', linewidth=3,
                 label=f"Linear Fit\n$D = {D_extracted:.4e}$ nm$^2$/fs")

        plt.axvline(x=times_fs[cutoff_index], color='gray', linestyle=':', label='Fit Cutoff')
        plt.text(times_fs[cutoff_index // 2], np.max(msd_nm2) * 0.8, 'Ballistic\n(Non-Linear)', ha='center',
                 color='gray')
        plt.text(times_fs[cutoff_index + (len(times_fs) - cutoff_index) // 2], np.max(msd_nm2) * 0.2,
                 'Diffusive\n(Linear)', ha='center', color='red')

        plt.title("Macroscopic Diffusion Verification")
        plt.xlabel("Time (fs)")
        plt.ylabel("Mean Squared Displacement (nm$^2$)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("Macroscopic Diffusion Verification", dpi=300)
        plt.show()

    return D_extracted

@_accepts_multiband
def visualize_simulation(frames: list, dt: float, save_interval: int,
                         length_x: float, length_y: float, grid_x: int, grid_y: int,
                         right_panel_mode: str = 'qgrid', target_state: int=None,
                         q_Nx: int = None, q_Ny: int = None):
    """
    Animates the simulation history with Total Density on the left,
    and either a specific Q-slice or the total Q-grid distribution on the right
    :param frames: List of frames from completed simulation.
    :param dt: time step
    :param save_interval: interval at which frames are saved
    :param length_x: spatial length in x
    :param length_y: spatial length in y
    :param grid_x: discretization of field in x
    :param grid_y: discretization of field in y
    :param q_Nx:
    :param q_Ny:
    :return: visualization of the simulation
    """

    # Validation / Setup
    Nq = frames[0].shape[2] # Extract the number of momentum states

    if right_panel_mode == 'slice':
        if target_state is None or target_state < 0 or target_state >= Nq:
            raise ValueError(f"WARNING: Invalid target_state. Your simulation has {Nq} momentum slices. "
                             f"Please specify a target_state between 0 and {Nq - 1}.")

    elif right_panel_mode == 'qgrid':
        # Auto-calculate Q-grid dimenstions assuming a square grid if not provided
        if q_Nx is None or q_Ny is None:
            q_Nx = int(np.sqrt(Nq))
            q_Ny = Nq // q_Nx
            if q_Nx * q_Ny != Nq:
                raise ValueError("could not auto-determine Q-grid dimensions. Please provide q_Nx and q_Ny.")

    else:
        raise ValueError("right_panel_mode must be either 'qgrid' or 'slice'.")


    # Recreate Spatial Grids
    delta_x = length_x / (grid_x - 1)
    delta_y = length_y / (grid_y - 1)

    X_grid = np.empty((grid_x, grid_y))
    Y_grid = np.empty((grid_x, grid_y))
    for i in range(grid_x):
        for j in range(grid_y):
            X_grid[i, j] = i * delta_x
            Y_grid[i, j] = j * delta_y

    X_flat = X_grid.flatten()
    Y_flat = Y_grid.flatten()

    # Initialize Plotting
    physical_time_per_frame = dt * save_interval

    plt.ion()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    colorbar_created = False

    # Animation Loop
    for i, frame in enumerate(frames):
        ax1.clear()
        ax2.clear()

        current_time_fs = i * physical_time_per_frame

        # Left Panel: Total PHysical Density
        total_density = np.sum(frame, axis=2)
        total_flat = total_density.flatten()

        sc1 = ax1.scatter(X_flat, Y_flat, c=total_flat, cmap='magma', marker='s', s=15)

        ax1.set_title(f"Total Exciton Density | Time: {current_time_fs:.3f} fs")
        ax1.set_xlabel("X Position (nm)")
        ax1.set_ylabel("Y Position (nm)")
        ax1.set_xlim(0, length_x)
        ax1.set_ylim(0, length_y)

        # Right Panel: qgrid or qslice
        if right_panel_mode == 'slice':
            slice_density = frame[:, :, target_state]
            slice_flat = slice_density.flatten()

            sc2 = ax2.scatter(X_flat, Y_flat, c=slice_flat, cmap='viridis', marker='s', s=15)
            ax2.set_title(f"Phase Space Slice [State {target_state}]")
            ax2.set_xlabel("X Position (nm)")
            ax2.set_ylabel("Y Position (nm)")
            ax2.set_xlim(0, length_x)
            ax2.set_ylim(0, length_y)

        elif right_panel_mode == 'qgrid':
            momentum_distribution_1d = np.sum(frame, axis=(0, 1))
            momentum_distribution_2d = momentum_distribution_1d.reshape((q_Nx, q_Ny))

            sc2 = ax2.imshow(momentum_distribution_2d.T, origin='lower', cmap='plasma',
                             extent=[-np.pi, np.pi, -np.pi, np.pi], interpolation='nearest')

            ax2.set_title("Momentum Space (Q-Grid)")
            ax2.set_xlabel("qx")
            ax2.set_ylabel("qy")

        # Colorbars
        if not colorbar_created:
            cbar1 = fig.colorbar(sc1, ax=ax1, fraction=0.046, pad=0.04)
            cbar1.set_label("Total Mass")

            cbar2 = fig.colorbar(sc2, ax=ax2, fraction=0.046, pad=0.04)
            cbar2.set_label("Slice Mass" if right_panel_mode == 'slice' else "Q-State Mass")

            colorbar_created = True

        plt.draw()
        plt.pause(0.05)

    plt.ioff()
    plt.show()

@_accepts_multiband
def visualize_explicit_q_slice(frames: list, dt: float, save_interval: int,
                               length_x: float, length_y: float, grid_x: int, grid_y: int,
                               target_state: int, q_vectors: np.ndarray):
    """

    :param frames: List of frames from completed simulation.
    :param dt: time step
    :param save_interval: interval at which frames are saved
    :param length_x: spatial length in x
    :param length_y: spatial length in y
    :param grid_x: discretization of field in x
    :param grid_y: discretization of field in y
    :param target_state: q-state that is being visualized
    :param q_vectors: Qpts from xctph.h5
    :return: visualization of real-space density q-slice
    """
    # Boundary Validation
    Nq = frames[0].shape[2]
    if target_state is None or target_state < 0 or target_state >= Nq:
        raise ValueError(f"Error: Target state {target_state} is out of bounds. "
                         f"Please select an index between 0 and {Nq - 1}.")

    if len(q_vectors) != Nq:
        raise ValueError(f"Fatal: Mismatch between simulation momentum states ({Nq}) "
                         f"and provided Q-vectors list ({len(q_vectors)}).")

    # Extract Q-Coordinates for Laveling
    qx, qy, qz = q_vectors[target_state]
    coordinate_label = f"Q = ({qx:.3f}, {qy:.3f}, {qz: .3f})"
    print(f"Launching explicit Q-Slice Visualization for State {target_state} | {coordinate_label}")

    # Initialize Plotting
    physical_time_per_frame = dt * save_interval

    plt.ion()
    fig, ax = plt.subplots(figsize=(8, 7))
    colorbar_created = False

    # Animation Loop
    for i, frame in enumerate(frames):
        ax.clear()
        current_time_fs = i * physical_time_per_frame

        #Extract density from target slice
        slice_density = frame[:, :, target_state]

        im = ax.imshow(slice_density.T, origin='lower', cmap='viridis',
                       extent=[0, length_x, 0, length_y], interpolation='nearest')

        #Explicit Labeling Standard
        ax.set_title(f"Explicit Q-Slice: Index {target_state}\n{coordinate_label} | Time: {current_time_fs:.3f} fs")
        ax.set_xlabel("X Position (nm)")
        ax.set_ylabel("Y Position (nm)")

        if not colorbar_created:
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label("Exciton Density in State")
            colorbar_created = True

        plt.draw()
        plt.pause(0.05)

    plt.ioff()
    plt.show()

@_accepts_multiband
def visualize_projected_momentum(frames: list, dt: float, save_interval: int,
                                    length_x: float, length_y: float,
                                    q_vectors: np.ndarray,
                                    projection: tuple = ('x','y')):
    """
    :param frames: List of frames from completed simulation.
    :param dt: time step
    :param save_interval: interval at which frames are saved
    :param length_x: spatial length in x
    :param length_y: spatial length in y
    :param q_vectors: Qpts from xctph.h5
    :param projection: directions for 2D projection of Q-space
    :return: visualization of full density and projected momentum
    """
    # Axis Mapping & Validation
    axis_map = {'x': 0, 'y': 1, 'z': 2}
    if projection[0] not in axis_map or projection[1] not in axis_map:
        raise ValueError("Projection tuple must strictly contain 'x', 'y', or 'z'.")

    ax1_idx = axis_map[projection[0]]
    ax2_idx = axis_map[projection[1]]
    Nq = frames[0].shape[2]
    # Map the physical Q-Grid for heatmap
    unique_q1 = np.unique(q_vectors[:, ax1_idx])
    unique_q2 = np.unique(q_vectors[:, ax2_idx])

    # Calculate bin widths to properly center the pixels in imshow
    dq1 = (unique_q1[1] - unique_q1[0]) if len(unique_q1) > 1 else 1.0
    dq2 = (unique_q2[1] - unique_q2[0]) if len(unique_q2) > 1 else 1.0

    q1_min, q1_max = unique_q1.min() - dq1 / 2, unique_q1.max() + dq1 / 2
    q2_min, q2_max = unique_q2.min() - dq2 / 2, unique_q2.max() + dq2 / 2
    q_extent = [q1_min, q1_max, q2_min, q2_max]

    # Setup Plotting
    physical_time_per_frame = dt * save_interval
    plt.ion()
    fig, (ax_real, ax_mom) = plt.subplots(1, 2, figsize=(14, 6))
    colorbar_created = False

    # Animation Loop
    for i, frame in enumerate(frames):
        ax_real.clear()
        ax_mom.clear()
        current_time_fs = i * physical_time_per_frame

        # -- LEFT PANEL: Total Real-Space Density --
        total_density = np.sum(frame, axis=2)
        im_real = ax_real.imshow(total_density.T, origin='lower', cmap='magma',
                                 extent=[0, length_x, 0, length_y], interpolation='nearest')

        ax_real.set_title(f"Total Exciton Density | Time: {current_time_fs:.3f} fs")
        ax_real.set_xlabel("X Position (nm)")
        ax_real.set_ylabel("Y Position (nm)")

        # -- RIGHT PANEL: Truncated Momentum Space
        # Create 2D grid
        momentum_2d = np.zeros((len(unique_q1), len(unique_q2)))

        # Sum out the unobserved axis
        for q_idx in range(Nq):
            density_in_state = np.sum(frame[:, :, q_idx])
            val1 = q_vectors[q_idx, ax1_idx]
            val2 = q_vectors[q_idx, ax2_idx]

            # Find matrix indices for this state's coordinate
            idx1 = np.where(unique_q1 == val1)[0][0]
            idx2 = np.where(unique_q2 == val2)[0][0]

            momentum_2d[idx1, idx2] += density_in_state

        im_mom = ax_mom.imshow(momentum_2d.T, origin='lower', cmap='plasma',
                               extent=q_extent, interpolation='nearest')

        ax_mom.set_title(f"Momentum Space Projection ({projection[0]}-{projection[1]} Plane)")
        ax_mom.set_xlabel(f"q_{projection[0]}")
        ax_mom.set_ylabel(f"q_{projection[1]}")

        # Colorbars
        if not colorbar_created:
            cbar1 = fig.colorbar(im_real, ax=ax_real, fraction=0.046, pad=0.04)
            cbar1.set_label("Total Density")

            cbar2 = fig.colorbar(im_mom, ax=ax_mom, fraction=0.046, pad=0.04)
            cbar2.set_label(f"Projected Mass in Q-Plane")
            colorbar_created = True

        plt.draw()
        plt.pause(0.05)

    plt.ioff()
    plt.show()


# ======================================================================
# Multi-state analysis (Chunk 4): populations and state-resolved diffusion
# ======================================================================

def _state_density(frame_4d, band):
    """Momentum-integrated 2D density rho_a(x, y) = sum_Q f_a(x, y, Q)."""
    return np.sum(np.asarray(frame_4d[band], dtype=np.float64), axis=-1)


def extract_state_populations(frames, dt, save_interval, state_labels=None):
    """
    Population N_a(t) = sum_{x,y,Q} f_a for every band, streamed one frame at a time.
    :param frames: (n_frames, n_bands, Nx, Ny, Nq) array-like (ndarray or h5py dataset)
    :return: times_fs (n_frames,), populations (n_frames, n_bands), state_labels
    """
    n_frames, n_bands = len(frames), frames[0].shape[0]
    labels = state_labels or [f"State {b}" for b in range(n_bands)]
    times = np.arange(n_frames) * dt * save_interval
    pops = np.zeros((n_frames, n_bands))
    for i in range(n_frames):
        fr = frames[i]
        for b in range(n_bands):
            pops[i, b] = np.sum(np.asarray(fr[b], dtype=np.float64))
    return times, pops, labels


def _msd_from_density(rho, X, Y):
    """Total mass, center of mass and variance about the center of mass."""
    m = rho.sum()
    if m <= 0:
        return m, (np.nan, np.nan), np.nan
    xc, yc = (rho * X).sum() / m, (rho * Y).sum() / m
    msd = (rho * ((X - xc) ** 2 + (Y - yc) ** 2)).sum() / m
    return m, (xc, yc), msd


def extract_multistate_diffusion(frames, dt, save_interval, x_grid, y_grid,
                                 state_labels=None, fit_window_fs=None,
                                 cutoff_fraction=0.5):
    """
    State-resolved and coupled diffusion constants from the spread about the center of mass:
        MSD_a(t) = <|r - r_cm,a(t)|^2>_a,   D = (1/4) dMSD/dt   (2D).
    Series are computed for each band and for the coupled total rho_tot = sum_a rho_a.

    :param fit_window_fs: (t_min, t_max) in fs for the linear fit. If None, the last
        (1 - cutoff_fraction) of the trajectory is used. The ballistic->diffusive
        crossover time is only a rough estimate; verify against naphthalene literature.
    :return: dict with times_fs, labels (bands + 'Coupled'), msd (n_frames, n_series),
        com (n_frames, n_series, 2), pops, D (nm^2/fs), intercept, fit_mask
    """
    n_frames, n_bands = len(frames), frames[0].shape[0]
    labels = list(state_labels or [f"State {b}" for b in range(n_bands)]) + ["Coupled"]
    times = np.arange(n_frames) * dt * save_interval
    X = np.asarray(x_grid, dtype=np.float64)
    Y = np.asarray(y_grid, dtype=np.float64)

    n_series = n_bands + 1
    msd = np.full((n_frames, n_series), np.nan)
    com = np.full((n_frames, n_series, 2), np.nan)
    pops = np.zeros((n_frames, n_series))

    for i in range(n_frames):
        fr = frames[i]
        rhos = [_state_density(fr, b) for b in range(n_bands)]
        rhos.append(np.sum(rhos, axis=0))
        for k, rho in enumerate(rhos):
            m, c, v = _msd_from_density(rho, X, Y)
            pops[i, k], com[i, k], msd[i, k] = m, c, v

    if fit_window_fs is not None:
        mask = (times >= fit_window_fs[0]) & (times <= fit_window_fs[1])
    else:
        mask = np.arange(n_frames) >= int(n_frames * cutoff_fraction)

    D = np.full(n_series, np.nan)
    intercept = np.full(n_series, np.nan)
    for k in range(n_series):
        ok = mask & np.isfinite(msd[:, k])
        if ok.sum() >= 2:
            slope, icpt = np.polyfit(times[ok], msd[ok, k], 1)
            D[k], intercept[k] = slope / 4.0, icpt

    return dict(times_fs=times, labels=labels, msd=msd, com=com, pops=pops,
                D=D, intercept=intercept, fit_mask=mask)


def plot_state_populations(times, pops, labels, filename="population_dynamics.png",
                           show_plot=False):
    """Absolute populations and fractions of total vs time."""
    total = pops.sum(axis=1)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5))
    for b, lab in enumerate(labels):
        a1.plot(times, pops[:, b], 'o-', label=lab)
        a2.plot(times, 100 * pops[:, b] / total, 'o-', label=lab)
    a1.plot(times, total, 'k--', label='Total')
    a1.set_xlabel("Time (fs)"); a1.set_ylabel("Population (integrated mass)")
    a2.set_xlabel("Time (fs)"); a2.set_ylabel("Fraction of total (%)")
    a1.set_title("State populations"); a2.set_title("Population fractions")
    for a in (a1, a2):
        a.grid(alpha=0.3); a.legend()
    plt.tight_layout(); plt.savefig(filename, dpi=300)
    if show_plot: plt.show()
    plt.close(fig)


def plot_multistate_msd(result, filename="msd_diffusion_analysis.png", show_plot=False):
    """MSD about center of mass for each state + coupled total, with linear fits."""
    t, mask = result['times_fs'], result['fit_mask']
    fig, ax = plt.subplots(figsize=(8, 6))
    for k, lab in enumerate(result['labels']):
        line, = ax.plot(t, result['msd'][:, k], 'o-', label=f"{lab} MSD")
        if np.isfinite(result['D'][k]):
            fit = 4 * result['D'][k] * t[mask] + result['intercept'][k]
            ax.plot(t[mask], fit, '--', color=line.get_color(), lw=2.5,
                    label=f"{lab} fit: D = {result['D'][k]:.3e} nm$^2$/fs")
    if mask.any():
        ax.axvline(t[mask][0], color='gray', ls=':', label='Fit start')
    ax.set_xlabel("Time (fs)"); ax.set_ylabel("MSD about center of mass (nm$^2$)")
    ax.set_title("State-resolved diffusion"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(filename, dpi=300)
    if show_plot: plt.show()
    plt.close(fig)


def export_multistate_gif(frames, dt, save_interval, length_x, length_y, grid_x, grid_y,
                          q_vectors, state_labels=None, projection=('x', 'y'),
                          filename="gifs/multistate_diffusion.gif", fps=15):
    """
    Exports a 4-panel animated GIF showing state-resolved real space and momentum space
    dynamics for coupled multi-band simulations (e.g. S2 and S1).
    Row 0: State 0 (S2) Real Space & Momentum Space
    Row 1: State 1 (S1) Real Space & Momentum Space
    """
    print(f"Rendering 4-panel multi-state GIF to {filename}...")
    n_frames = len(frames)
    n_bands = frames[0].shape[0] if frames[0].ndim == 4 else 1
    labels = state_labels or [f"State {b}" for b in range(n_bands)]
    Nq = frames[0].shape[-1]
    
    axis_map = {'x': 0, 'y': 1, 'z': 2}
    ax1_idx = axis_map[projection[0]]
    ax2_idx = axis_map[projection[1]]
    unique_q1 = np.unique(q_vectors[:, ax1_idx])
    unique_q2 = np.unique(q_vectors[:, ax2_idx])
    dq1 = (unique_q1[1] - unique_q1[0]) if len(unique_q1) > 1 else 1.0
    dq2 = (unique_q2[1] - unique_q2[0]) if len(unique_q2) > 1 else 1.0
    q_extent = [unique_q1.min() - dq1 / 2, unique_q1.max() + dq1 / 2,
                unique_q2.min() - dq2 / 2, unique_q2.max() + dq2 / 2]

    # Precompute spatial and momentum histories for all bands
    spatial_history = {b: [] for b in range(n_bands)}
    mom_history = {b: [] for b in range(n_bands)}
    pop_history = {b: [] for b in range(n_bands)}
    
    for fr in frames:
        for b in range(n_bands):
            f_b = fr[b] if fr.ndim == 4 else fr
            # Real space density: sum over Q
            rho = np.sum(f_b, axis=-1)
            spatial_history[b].append(rho)
            pop = np.sum(rho)
            pop_history[b].append(pop)
            
            # Momentum density: sum over x, y
            mass_per_q = np.sum(f_b, axis=(0, 1))
            m_2d = np.zeros((len(unique_q1), len(unique_q2)))
            for q_idx in range(Nq):
                v1, v2 = q_vectors[q_idx, ax1_idx], q_vectors[q_idx, ax2_idx]
                i1 = np.where(unique_q1 == v1)[0][0]
                i2 = np.where(unique_q2 == v2)[0][0]
                m_2d[i1, i2] += mass_per_q[q_idx]
            mom_history[b].append(m_2d)

    # Global color limits across all frames and states
    max_spatial = max(np.max(spatial_history[b]) for b in range(n_bands))
    max_mom = max(np.max(mom_history[b]) for b in range(n_bands))

    fig, axes = plt.subplots(n_bands, 2, figsize=(12, 5 * n_bands))
    if n_bands == 1:
        axes = np.array([axes])

    im_spatial = []
    im_mom = []

    for b in range(n_bands):
        ax_sp, ax_mom = axes[b, 0], axes[b, 1]
        
        im_s = ax_sp.imshow(spatial_history[b][0].T, origin='lower', cmap='magma',
                            extent=[0, length_x, 0, length_y], vmin=0, vmax=max_spatial)
        ax_sp.set_xlabel("X (nm)")
        ax_sp.set_ylabel("Y (nm)")
        fig.colorbar(im_s, ax=ax_sp, fraction=0.046, pad=0.04).set_label("Exciton Mass")
        im_spatial.append(im_s)
        
        im_m = ax_mom.imshow(mom_history[b][0].T, origin='lower', cmap='plasma',
                            extent=q_extent, vmin=0, vmax=max_mom)
        ax_mom.set_xlabel(f"q_{projection[0]}")
        ax_mom.set_ylabel(f"q_{projection[1]}")
        fig.colorbar(im_m, ax=ax_mom, fraction=0.046, pad=0.04).set_label("Occupancy")
        im_mom.append(im_m)

    def update(frame_idx):
        t_fs = frame_idx * dt * save_interval
        total_p = sum(pop_history[b][frame_idx] for b in range(n_bands))
        artists = []
        for b in range(n_bands):
            p_b = pop_history[b][frame_idx]
            pct = (p_b / total_p * 100.0) if total_p > 0 else 0.0
            
            im_spatial[b].set_data(spatial_history[b][frame_idx].T)
            axes[b, 0].set_title(f"[{labels[b]}] Real Space | Pop: {pct:.1f}% | t = {t_fs:.2f} fs")
            
            im_mom[b].set_data(mom_history[b][frame_idx].T)
            axes[b, 1].set_title(f"[{labels[b]}] Momentum Space (q_{projection[0]}-q_{projection[1]})")
            
            artists.extend([im_spatial[b], im_mom[b]])
        return artists

    ani = animation.FuncAnimation(fig, update, frames=n_frames, blit=False)
    ani.save(filename, writer=PillowWriter(fps=fps))
    plt.close(fig)
    print(f"Multi-state GIF saved successfully to '{filename}'!")

