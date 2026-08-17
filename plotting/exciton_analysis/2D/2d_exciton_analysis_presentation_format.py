import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import ListedColormap
import matplotlib.ticker as ticker
import pathlib


def plot_scatter_heatmap(filename, save_plot=False, marker_size=15,
                         x_label="a", y_label="b",
                         cell_vectors=None, supercell_dims=None):
    """
    Loads 3-column (x, y, z) fractional data, dynamically computes nanometer
    scaling factors from provided cell vectors and supercell dimensions,
    centers the hole at (0,0), and plots a 2D ECF heatmap.
    """
    try:
        # --- 1. Load Data ---
        print(f"Loading data from {filename}...")
        data = np.loadtxt(filename, comments='#')

        if data.ndim == 1 or data.shape[1] < 3:
            print("Error: Data must have at least 3 columns (x, y, z).")
            return

        # --- 2. Dynamic Unit Conversion & Centering ---
        # Ensure default fallback dictionaries exist if none are provided
        # Pentacene Default
        if cell_vectors is None:
            raise ValueError("Cell Parameters Needed.")

        if supercell_dims is None:
            supercell_dims = {'a': 8, 'b': 8, 'c': 4}

        # Calculate the magnitude (length) of the requested cell vectors in Angstroms
        mag_x_angstrom = np.linalg.norm(cell_vectors[x_label])
        mag_y_angstrom = np.linalg.norm(cell_vectors[y_label])

        # Calculate the total physical length of the supercell axes in nanometers
        # (Multiply by supercell dimension, then divide by 10 to convert A -> nm)
        len_x_nm = (mag_x_angstrom * supercell_dims[x_label]) / 10.0
        len_y_nm = (mag_y_angstrom * supercell_dims[y_label]) / 10.0

        # Apply transformation: Center the grid (subtract 1.0) and scale to nm
        x = (data[:, 0] - 1.0) * len_x_nm
        y = (data[:, 1] - 1.0) * len_y_nm

        # Keep the true z-correlation values
        z_correlation = data[:, 2]

        file_stem = pathlib.Path(filename).stem

        # --- 3. Plotting ---
        print("Generating scatter heatmap...")

        # Large square figure size for presentation slides
        fig, ax = plt.subplots(figsize=(10, 10))

        # Set ONLY the plot area background to black (leaves outer figure white)
        ax.set_facecolor('black')

        # --- Build the Custom Colormap ---
        # Get 11 stepped colors from the standard jet colormap
        jet_colors = cm.get_cmap('jet', 11)(np.linspace(0, 1, 11))
        # Define pure black [R, G, B, Alpha]
        black = np.array([[0.0, 0.0, 0.0, 1.0]])
        # Stack black at the bottom of the jet colors
        custom_colors = np.vstack((black, jet_colors))
        # Generate the new 12-step discrete colormap
        discrete_cmap = ListedColormap(custom_colors)

        # Create the scatter plot without marker edges to prevent scaling artifacts
        scatter = ax.scatter(x, y, c=z_correlation, cmap=discrete_cmap,
                             s=marker_size, edgecolors='none')

        # Restored labels since the axes now represent true physical distances
        ax.set_xlabel('Electron-hole distance (nm)', fontsize=35, fontweight='bold', color='black')
        ax.set_ylabel('Electron-hole distance (nm)', fontsize=35, fontweight='bold', color='black')

        # Place the crystallographic plane label above the plot
        ax.set_title(f"({x_label}{y_label})", fontsize=40, pad=20, color='black')

        # --- Tick Mark Configuration ---
        # Set the numbered labels (major ticks) to appear every 1.0 unit
        ax.xaxis.set_major_locator(ticker.MultipleLocator(1.0))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(1.0))

        # Set the unnumbered tick marks (minor ticks) to appear every 0.5 units
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.5))
        ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.5))

        # Apply styling to BOTH major and minor tick lines
        ax.tick_params(axis='both', which='both', direction='out', length=8, width=2, colors='black')

        # Apply the large font size ONLY to the major tick labels
        ax.tick_params(axis='both', which='major', labelsize=20)

        # Ensure axes scales are identical to preserve true physical distances
        ax.set_aspect('equal')

        # Add the stepped color bar
        cbar = fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=18, colors='black')

        # Remove grid lines for a cleaner look
        ax.grid(False)

        # Frame the ECF centrally identically to the literature bounds
        ax.set_xlim([-2.0, 2.0])
        ax.set_ylim([-2.0, 2.0])

        # --- 4. Show Plot and/or Save Plot ---
        if save_plot:
            # Save at 600 DPI to ensure high resolution when scaled down on slides
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


# --- How to use the function ---
if __name__ == "__main__":
    file_to_plot = "data/63cm-1_300K/plane_samp_63cm-1_2D-ab.gp"
    save_plot = True

    # Define the fundamental cell vectors in Angstroms
    input_vectors = {
        'a': np.array([6.266000000000000, 0.000000000000000, 0.000000000000000]),
        'b': np.array([0.7203431964649767, 7.7415586724707204, 0.000000000000000]),
        'c': np.array([0.5876759734010517, 3.3581219057453895, 14.1243957115495764])
    }

    # Define the supercell grid size
    input_supercell = {'a': 8, 'b': 8, 'c': 4}

    plot_scatter_heatmap(file_to_plot,
                         save_plot=save_plot,
                         marker_size=15,
                         x_label="a",
                         y_label="b",
                         cell_vectors=input_vectors,
                         supercell_dims=input_supercell)