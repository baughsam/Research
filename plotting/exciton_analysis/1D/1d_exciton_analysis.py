import numpy as np
import matplotlib.pyplot as plt
import pathlib


def plot_dat_file(filename, axis):
    """
    Loads and plots data from a 2-column .ddat file.
    Assumes the first line(s) might be comments starting with '#'.
    """
    try:
        # Load the data from the file
        data = np.loadtxt(filename, comments='#')

        # Extract the columns
        x = data[:, 0]
        y = data[:, 1]

         # --- Create the Plot ---
        plt.figure(figsize=(10, 6))

        # Plot the data
        plt.plot(x, y)  # 'o-' = line plot with circle markers

        # --- Get just the filename for the title ---
        # pathlib.Path(filename).name extracts the file part (e.g., "ex_pristine_1D-a.dat")
        plot_title = pathlib.Path(filename).name

        # Add labels and a title
        plt.xlabel(f"Position on {axis}")
        plt.ylabel("averaged 2part-corr")
        plt.title(f"Plot of {plot_title} data")

        # Add a grid for easier reading
        #plt.grid(True)

        # Save the plot
        plt.savefig(f"./plots/{plot_title}.png")

        # Display the plot
        plt.show()

    except FileNotFoundError:
        print(f"Error: The file '{filename}' was not found.")
    except Exception as e:
        print(f"An error occurred while processing the file: {e}")
        print("Please check that the file is formatted correctly.")


# --- How to use the function ---

# 1. Set the name of your data file
file_to_plot = "data/prist_12x12x6_samp55_hole_pos/plane_samp_dir+ex_pristine_1D-c.dat"


# 2. Call the function to plot it
plot_dat_file(file_to_plot, axis="c")