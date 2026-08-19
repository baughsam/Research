import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import ListedColormap
import matplotlib.ticker as ticker
import pathlib


def plot_scatter_heatmap(filename, save_plot=False, marker_size=15,
                         x_label="a", y_label="b"):
    """
    Loads 3-column (x, y, z) data in Bohr, converts to nanometers,
    and plots a high-contrast 2D ECF scatter heatmap for presentations.
    """
    try:
        # --- 1. Load Data ---
        print(f"Loading data from {filename}...")
        data = np.loadtxt(filename, comments='#')

        if data.ndim == 1 or data.shape[1] < 3:
            print("Error: Data must have at least 3 columns (x, y, z).")
            return

        # --- 2. Direct Conversion from Bohr to Nanometers ---
        BOHR_TO_NM = 0.0529177210903

        # Raw data is already centered in Bohr; scale directly to nm
        x = data[:, 0] * BOHR_TO_NM
        y = data[:, 1] * BOHR_TO_NM
        z_correlation = data[:, 2]

        file_stem = pathlib.Path(filename).stem

        # --- 3. Plotting ---
        print("Generating scatter heatmap...")

        fig, ax = plt.subplots(figsize=(10, 10))
        ax.set_facecolor('black')

        # Custom discrete colormap with pure black background
        jet_colors = cm.get_cmap('jet', 11)(np.linspace(0, 1, 11))
        black = np.array([[0.0, 0.0, 0.0, 1.0]])
        discrete_cmap = ListedColormap(np.vstack((black, jet_colors)))

        scatter = ax.scatter(x, y, c=z_correlation, cmap=discrete_cmap,
                             s=marker_size, edgecolors='none')

        # Axis and Title Labels
        ax.set_xlabel('Electron-hole distance (nm)', fontsize=26, fontweight='bold', color='black')
        ax.set_ylabel('Electron-hole distance (nm)', fontsize=26, fontweight='bold', color='black')
        ax.set_title(f"({x_label}{y_label})", fontsize=32, pad=15, color='black')

        # Set major ticks every 1.0 or 2.0 nm, minor every 0.5 nm
        ax.xaxis.set_major_locator(ticker.MultipleLocator(2.0))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(2.0))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.5))
        ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.5))

        ax.tick_params(axis='both', which='both', direction='out', length=8, width=2, colors='black')
        ax.tick_params(axis='both', which='major', labelsize=18)

        ax.set_aspect('equal')

        # Explicit physical bounds in nm
        max_bound = max(np.max(np.abs(x)), np.max(np.abs(y)))
        ax.set_xlim([-2.0, 2.0])
        ax.set_ylim([-2.0, 2.0])

        # Colorbar
        cbar = fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=16, colors='black')

        ax.grid(False)

        # --- 4. Show / Save Plot ---
        if save_plot:
            save_filename = f"{file_stem}_scatter_heatmap.png"
            out_path = pathlib.Path("./plots")
            out_path.mkdir(exist_ok=True)
            plt.savefig(out_path / save_filename, dpi=600, bbox_inches='tight')
            print(f"Plot saved successfully to: {out_path / save_filename}")

        plt.show()

    except FileNotFoundError:
        print(f"Error: The file '{filename}' was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    file_to_plot = "data/plane_samp_dir+ex_prist_trip_2D-ab_density_bohr.dat"
    plot_scatter_heatmap(file_to_plot, save_plot=True, marker_size=12, x_label="a", y_label="b")