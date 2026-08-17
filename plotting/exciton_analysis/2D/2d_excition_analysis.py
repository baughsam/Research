import numpy as np
import matplotlib.pyplot as plt
import pathlib


def plot_scatter_heatmap(filename, save_plot=False, marker_size=10, x_label ="1", y_label = "2"):
    """
    Loads 3-column (x, y, z) data and plots it as a 2D scatter plot,
    where the color of each point is determined by the z-value.

    Args:
        filename (str): The path to the data file.
        save_plot (bool): If True, saves the plot to a .png file.
        marker_size (int): The size of the markers in the scatter plot.
    """

    try:
        # --- 1. Load Data ---
        print(f"Loading data from {filename}...")
        data = np.loadtxt(filename, comments='#')

        if data.ndim == 1 or data.shape[1] < 3:
            print("Error: Data must have at least 3 columns (x, y, z).")
            return

        x = data[:, 0]
        y = data[:, 1]
        z_correlation = data[:, 2]  # Correlation

        # Get a clean filename stem (e.g., "my_data" from "path/to/my_data.dat")
        file_stem = pathlib.Path(filename).stem
        plot_title_name = pathlib.Path(filename).name

        # --- 2. Plotting ---
        print("Generating scatter heatmap...")

        fig, ax = plt.subplots(figsize=(12, 8))

        # Create the scatter plot
        # x = Position 1
        # y = Position 2
        # c = Color based on z_correlation
        # cmap = Colormap (viridis is a good default)
        # s = Marker size
        scatter = ax.scatter(x, y, c=z_correlation, cmap='viridis', s=marker_size)

        # Add labels and title
        ax.set_xlabel(f'Position {x_label}')
        ax.set_ylabel(f'Position {y_label}')
        ax.set_title(f"Scatter Heatmap of {plot_title_name}")

        # Add a color bar to show the mapping of z-values to color
        cbar = fig.colorbar(scatter, ax=ax, label='Correlation')

        # Set aspect ratio to be equal
        ax.set_aspect('equal')
        ax.grid(True, linestyle='--', alpha=0.6)

        # --- 3. Show Plot and/or Save Plot ---
        if save_plot:
            # Generate a descriptive save name
            save_filename = f"{file_stem}_scatter_heatmap.png"
            plt.savefig(f"./plots/{save_filename}", dpi=300, bbox_inches='tight')
            print(f"Plot saved successfully to: {save_filename}")

        # Show the plot
        plt.show()

    except FileNotFoundError:
        print(f"Error: The file '{filename}' was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")


# --- How to use the function ---

# 1. Set the name of your data file
file_to_plot = "data/envelope_cx_100perc_S3_91cm-1_2D-bc.gp"  # <-- Change this

# 2. Set whether to save the plot: True or False
save_plot = True  # <-- Set this to True to save the file

# 3. Call the function
plot_scatter_heatmap(file_to_plot,
                     save_plot=save_plot,
                     marker_size=1,
                     x_label="b",
                     y_label="c")  # You can adjust marker_size