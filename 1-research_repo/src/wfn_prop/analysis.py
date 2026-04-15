import numpy as np
import matplotlib.pyplot as plt


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


