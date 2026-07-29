import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter


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



def export_diffusion_gif_updated(frames, dt, save_interval, length_x, length_y, grid_x, grid_y,
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