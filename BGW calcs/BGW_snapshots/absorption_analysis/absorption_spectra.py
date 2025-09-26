import numpy as np
import matplotlib.pyplot as plt
import os

# Define the folder containing the data files
folder_name = "absorption_files"

# --- Plotting Setup ---
# Create a figure and axes for the plot
plt.figure(figsize=(8, 6))

# Get a colormap. 'viridis', 'plasma', or 'jet' are good choices for 10 lines.
# We will generate 10 distinct colors from this map.
num_files = 10
colors = plt.cm.viridis(np.linspace(0, 1, num_files))

absorp_dir = "b3" #which direction (b1, b2, or b3)


# --- Data Loading and Plotting Loop ---
# Loop through the file numbers from 1 to 10
for k in range(1, num_files + 1):
    # Construct the filename using an f-string
    file_name = f"absorption_{absorp_dir}_eh_struct_{k}.dat"
    # Create the full path to the file
    file_path = os.path.join(folder_name, file_name)
    print(file_path)

    # Check if the file actually exists before trying to plot it
    if os.path.exists(file_path):
        # Load the data from the file
        data = np.loadtxt(file_path, comments='#')
        omega = data[:, 0]  # Energy in eV
        eps2 = data[:, 1]   # Absorption intensity

        # Plot the data with a specific color, thin linewidth, and a label
        plt.plot(omega, eps2, color=colors[k-1], linewidth=1, label=f'b{k}')
    else:
        print(f"Warning: File not found and will be skipped: {file_path}")

# --- Final Plot Customizations ---
plt.xlabel("Energy (eV)")
plt.ylabel(r"Absorption Intensity ($\epsilon_2$)") # Using LaTeX for epsilon
plt.ylim(0, 3.25) # Adjust if your data has a different range
plt.xlim(1,4) # Adjust to the energy range of interest
plt.title(f'Absorption Spectra (1 - {num_files}) ({absorp_dir})')
plt.legend(title="Structure") # Adds a legend to identify the lines
plt.grid(True, linestyle='--', alpha=0.6) # Add a light grid for readability
plt.tight_layout() # Adjusts plot to ensure everything fits without overlapping

# --- Save and Display the Plot ---
plt.savefig(f'absorption_spectra_comparison_{absorp_dir}.png', dpi=300) # Save with high resolution
plt.show()