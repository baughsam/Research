import numpy as np
import matplotlib.pyplot as plt
import sys

# ==========================================
# 1. CONFIGURATION SECTION
# ==========================================

# Define your datasets here.
# Each block inside [] is one "Comparison".
# You can add as many blocks as you need.
# Ensure 'files' always has 3 entries: [b1, b2, b3]

#source directory of data
src_dir ="./data/"
datasets = [
    {
        "label": "Pristine",
        "files": [f"{src_dir}absorption_b1_eh_pent_prist.dat",
                  f"{src_dir}absorption_b2_eh_pent_prist.dat",
                  f"{src_dir}absorption_b3_eh_pent_prist.dat"]
    },
    {
        "label": "T-Dep (Frozen Phonon)",
        "files": [f"{src_dir}absorption_b1_eh_100_tdep.dat",
                  f"{src_dir}absorption_b2_eh_100_tdep.dat",
                  f"{src_dir}absorption_b3_eh_100_tdep.dat"]
    },

   # Example: You can easily add different data files to compare more than two systems
   # {
   #     "label": "Future Comparison (e.g. 300K)",
   #     "files": ["absorption_b1_300k.dat",
   #               "absorption_b2_300k.dat",
   #               "absorption_b3_300k.dat"]
   # },
]

# Plotting Settings
# The script will cycle through these colors in order. You can change them to what you want
color_cycle = ['black', 'red', 'blue', 'green', 'orange', 'purple', 'cyan']
xlim_range = (0, 4.0)  # eV range for x-axis
ylim_range = (0, 3.5) #epsilon range for y-axis
line_width = 1.5


# ==========================================
# 2. DATA LOADING & PROCESSING
# ==========================================

def load_file(fname):
    """Loads (energy, eps2) from a file. Returns None on failure."""
    try:
        data = np.loadtxt(fname, comments='#')
        return data[:, 0], data[:, 1]
    except Exception as e:
        print(f"Error loading {fname}: {e}")
        return None, None


# Dictionary to hold processed data for plotting
# Structure: plot_data[dataset_index] = [ (x, y_b1), (x, y_b2), (x, y_b3), (x, y_avg) ]
processed_data = {}
first_energy_grid = None

print(f"Processing {len(datasets)} datasets...")

for i, ds in enumerate(datasets):
    files = ds['files']
    current_eps2_stack = []
    current_plot_lines = []  # Will hold the 4 lines (b1, b2, b3, avg) for this dataset

    # Load b1, b2, b3
    for fname in files:
        energy, eps2 = load_file(fname)

        if energy is None:
            print("Critical Error: Missing file. Exiting.")
            sys.exit()

        # Grab energy grid from the very first file loaded to ensure consistency
        if first_energy_grid is None:
            first_energy_grid = energy

        current_eps2_stack.append(eps2)
        current_plot_lines.append(eps2)  # Add b1, b2, b3 to list

    # Calculate Average
    # Stack into shape (3, N_points) and take mean
    stack_array = np.vstack(current_eps2_stack)
    avg_eps2 = np.mean(stack_array, axis=0)
    current_plot_lines.append(avg_eps2)  # Add Average to list (index 3)

    processed_data[i] = current_plot_lines

# ==========================================
# 3. PLOTTING
# ==========================================

panel_titles = ["b1 Direction", "b2 Direction", "b3 Direction", "Average"]
panel_indices = [0, 1, 2, 3]  # Corresponds to the indices in current_plot_lines

# --- A. Combined 4-Panel Plot ---
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

for panel_idx, ax in enumerate(axes):
    title = panel_titles[panel_idx]

    # Loop through all datasets (Pristine, T-Dep, etc.) and add them to this panel
    for ds_idx, ds in enumerate(datasets):
        y_values = processed_data[ds_idx][panel_idx]
        c = color_cycle[ds_idx % len(color_cycle)]  # Cycle colors if > 7 datasets #we've 7 preset colors, add more for no cycling
        lbl = ds['label']

        # Use dashed line for anything after the first one (optional style choice)
        ls = '-' if ds_idx == 0 else '--'

        ax.plot(first_energy_grid, y_values, color=c, label=lbl, lw=line_width, linestyle=ls)

    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel("Energy (eV)")
    ax.set_ylabel(r"$\varepsilon_2(\omega)$")
    ax.set_xlim(xlim_range)
    ax.set_ylim(ylim_range)
    if panel_idx == 0:  # Only show legend on the first plot to avoid clutter? Or all?
        ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("Comparison_Combined_Multi.png", dpi=300)
print("Saved combined plot: Comparison_Combined_Multi.png")
plt.show()

# --- B. Individual Plots ---
# We generate 4 separate image files (b1, b2, b3, avg)
for panel_idx, title in zip(panel_indices, panel_titles):
    plt.figure(figsize=(6, 5))

    for ds_idx, ds in enumerate(datasets):
        y_values = processed_data[ds_idx][panel_idx]
        c = color_cycle[ds_idx % len(color_cycle)]
        lbl = ds['label']
        ls = '-' if ds_idx == 0 else '--'

        plt.plot(first_energy_grid, y_values, color=c, label=lbl, lw=line_width, linestyle=ls)

    plt.title(title, fontsize=14)
    plt.xlabel("Energy (eV)")
    plt.ylabel(r"$\varepsilon_2(\omega)$")
    plt.xlim(xlim_range)
    plt.ylim(ylim_range)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    safe_name = title.split()[0]  # b1, b2, b3, Average
    filename = f"Comparison_{safe_name}.png"
    plt.savefig(filename, dpi=300)
    print(f"Saved individual plot: {filename}")
    plt.close()