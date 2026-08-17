#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import pathlib
import sys

# ==========================================
# 1. CONFIGURATION
# ==========================================

# Define the Comparisons you want to make.
# Each dictionary in this list represents one "Condition" (e.g. Pristine, T-Dep).
# The script will plot these curves together on the same graph for a given direction.

#Colors
colors = [
    '#5E4FA2', '#3288BD', '#66C2A5', '#ABDDA4', '#E6F598',
    '#FFFFBF', '#FEE08B', '#FDAE61', '#F46D43', '#9E0142'
]

src_dir_data = "./data/"

conditions = [
{
        "label": "Pristine",
        "color": "black",
        "linestyle": "-",
        "files": {
            "a": f"{src_dir_data}pristine/envelope_cx_pristine_1D-a.dat",
            "b": f"{src_dir_data}pristine/envelope_cx_pristine_1D-b.dat",
            "c": f"{src_dir_data}pristine/envelope_cx_pristine_1D-c.dat"
        }
    },
{
        "label": "91cm-1 300K S1",
        "color": "blue",
        "linestyle": "-",
        "files": {
            "a": f"{src_dir_data}91cm-1_300K/envelope_91cm-1_300K_1D-a.dat",
            "b": f"{src_dir_data}91cm-1_300K/envelope_91cm-1_300K_1D-b.dat",
            "c": f"{src_dir_data}91cm-1_300K/envelope_91cm-1_300K_1D-c.dat"
        }
    },
{
        "label": "91cm-1 300K S2",
        "color": "red",
        "linestyle": "-",
        "files": {
            "a": f"{src_dir_data}envelope_cx_100perc_S2_91cm-1_1D-a.dat",
            "b": f"{src_dir_data}envelope_cx_100perc_S2_91cm-1_1D-b.dat",
            "c": f"{src_dir_data}envelope_cx_100perc_S2_91cm-1_1D-c.dat"
        }
    },

{
        "label": "91cm-1 300K S3",
        "color": "green",
        "linestyle": "-",
        "files": {
            "a": f"{src_dir_data}envelope_cx_100perc_S3_91cm-1_1D-a.dat",
            "b": f"{src_dir_data}envelope_cx_100perc_S3_91cm-1_1D-b.dat",
            "c": f"{src_dir_data}envelope_cx_100perc_S3_91cm-1_1D-c.dat"
        }
    },

]

# Directions to plot (Must match keys in the 'files' dictionary above)
directions_to_plot = ["a", "b", "c"]

#plot source directory
src_dir_plots = "./plots/"

# Output Directory
output_dir = f"./plots/1d_exciton_comparison_Pris_91cm-1-S1_91cm-1-S2"

# Plot Styling
figure_size = (8, 6)
line_width = 0.5
font_size = 14
grid_on = True


# ==========================================
# 2. DATA LOADING
# ==========================================

def load_data(filename):
    """
    Loads data from a 2-column .dat file.
    Returns x, y or None, None if failed.
    """
    try:
        data = np.loadtxt(filename, comments='#')
        if data.ndim == 1:
            # Handle case where there might be only one point or weird formatting
            return np.array([data[0]]), np.array([data[1]])
        return data[:, 0], data[:, 1]
    except Exception as e:
        print(f"Warning: Could not load '{filename}'. Reason: {e}")
        return None, None


# ==========================================
# 3. PLOTTING LOGIC
# ==========================================

# Create output directory if it doesn't exist
pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

print(f"--- Generating {len(directions_to_plot)} comparison plots ---")

# Loop over each direction (e.g., "a", "b", "c")
for direction in directions_to_plot:
    print(f"Processing direction: {direction}...")

    # Initialize Plot for this direction
    fig, ax = plt.subplots(figsize=figure_size)

    has_data = False

    # Loop over each condition (Pristine, T-Dep, etc.)
    for cond in conditions:
        label = cond["label"]
        color = cond.get("color", None)  # Auto-color if None
        ls = cond.get("linestyle", "-")

        # Get filename for current direction
        fname = cond["files"].get(direction)

        if fname:
            x, y = load_data(fname)

            if x is not None:
                ax.plot(x, y, label=label, color=color, linestyle=ls, linewidth=line_width)
                has_data = True
            else:
                print(f"  -> Missing data for {label} in direction {direction}")
        else:
            print(f"  -> No filename specified for {label} in direction {direction}")

    # Formatting
    if has_data:
        ax.set_xlabel(f"Position along {direction}-axis", fontsize=font_size)
        ax.set_ylabel("Averaged 2-Particle Correlation", fontsize=font_size)
        ax.set_title(f"Exciton Wavefunction Comparison ({direction}-direction)", fontsize=font_size + 2)

        if grid_on:
            ax.grid(True, alpha=0.3)

        ax.legend(fontsize=font_size - 2)

        # Save
        save_path = f"{output_dir}/Exciton_1D_Comparison_{direction}.png"
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        print(f"  -> Saved: {save_path}")
        plt.close()
    else:
        print(f"  -> Skipped plot for {direction} (No data found).")
        plt.close()

print("Done.")