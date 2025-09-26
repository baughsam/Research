import pandas as pd
import matplotlib.pyplot as plt

#Comparison plot
#Data is taken from a .csv file (see ./bandgap_files for example on how data should be formatted)

# Load the data from the CSV file
# This line will work once the file is successfully uploaded.
df = pd.read_csv('./bandgap_files/BGW_snapshots_bandgaps.csv')

# Create a new column with the desired x-axis labels
df['struct_labels'] = 'struct_' + df['struct'].astype(str)

# Create the plot with a larger size for better readability
plt.figure(figsize=(14, 10))

# Plot VBM and CBM lines using the new labels for the x-axis
plt.plot(df['struct_labels'], df['VBM'], marker='o', linestyle='-', label='VBM')
plt.plot(df['struct_labels'], df['CBM'], marker='o', linestyle='-', label='CBM')

# --- Label Position Adjustment ---
# Adjust this value to move the labels.
# Positive values move the labels up, negative values move them down.
label_offset = -0.15
# --------------------------------

# --- Title ---
# Change title so that it is unique to your project
project_title = "BGW_snapshots"
# --------------------------------

# Add labels, title, and legend
plt.xlabel('Structure')
plt.ylabel('Energy (eV)')
plt.title(f'VBM and CBM Values for Each Structure ({project_title})')
plt.legend()
plt.grid(True)

# Rotate x-axis labels for better readability
plt.xticks(rotation=45, ha='right')

# Annotate the band gap for each structure
for i in range(len(df)):
    gap = df['CBM'][i] - df['VBM'][i]
    midpoint = (df['VBM'][i] + df['CBM'][i]) / 2
    plt.text(df['struct_labels'][i], midpoint + label_offset, f'{gap:.3f} eV', ha='center', va='center', bbox=dict(facecolor='white', alpha=0.5, boxstyle='round,pad=0.2'))

# Adjust layout to prevent labels from being cut off
plt.tight_layout()

# Save the plot to a file
plt.savefig(f'vbm_cbm_plot_{project_title}.png')

# To display the plot in a script, you would use:
plt.show()