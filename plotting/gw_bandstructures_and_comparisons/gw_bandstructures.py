import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# ==========================================
# 1. CONFIGURATION
# ==========================================

# A. DATASETS
# Add as many files as you want here.


src_dir = "./data/"
datasets = [
    {
        "label": "Pristine",
        "file": f"{src_dir}bandstructure_prist.dat",
        "color": "black"
    },
    {
        "label": "100% T-Dep (91cm-1)",
        "file": f"{src_dir}bandstructure_100_tdep.dat",
        "color": "blue"
    },

    #Example: Easily adding mother systems to compare with
    # { "label": "System 3", "file": "bandstructure_sys3.dat" },
]



# --- PHYSICS PARAMETERS ---
# Band Index of the VBM (Valence Band Maximum).
# IMPORTANT: In your v3 script, this was 8. For full Pentacene, it is 51.
# If your plot looks shifted/wrong, CHANGE THIS NUMBER.
nv = 102

# --- TOGGLES ---
plot_lda = False  # Set to True to plot LDA lines (dotted)

# --- PLOTTING RANGES ---
# Energy window (eV) relative to VBM
emin, emax = 2, 5.5

# --- OUTPUT ---
output_filename = "band_comparison_final.png"

# --- STYLING ---
band_linewidth = 1.0  # Main band lines
helper_linewidth = 0.5  # Grid/Fermi lines
lda_alpha = 0.7  # Transparency of LDA lines

# K-Point Path (Indices must match your file's k-grid)
k_special_index = np.array([0, 30, 60, 90, 120, 150, 180, 210])
k_special_label = np.array(['$\Gamma$', 'X', 'Y', '$\Gamma$', 'Z', 'U', 'R', 'Z'])


# ==========================================
# 2. DATA LOADING FUNCTION
# ==========================================
def load_bands(filepath, nv_bands):
    """
    Reads bands, finds VBM using Band Index 'nv_bands', aligns VBM to 0 eV.
    Returns aligned x, eqp, elda, gap_gw, gap_lda
    """
    try:
        data = np.loadtxt(filepath)
    except IOError:
        print(f"Error: File '{filepath}' not found.")
        return None, None, None, 0, 0

    # 1. Identify Bands
    bands_list = np.unique(data[:, 1])
    nb = len(bands_list)

    # 2. Extract K-points & X-axis
    cond_first_band = (data[:, 1] == bands_list[0])
    kpoints = data[cond_first_band][:, 2:5]
    nk = len(kpoints)

    dk = kpoints[1:] - kpoints[:-1]
    dk_norm = np.linalg.norm(dk, axis=1)
    dk_norm = np.insert(dk_norm, 0, 0)
    x_axis = np.cumsum(dk_norm)

    # 3. Extract Energies (Reshape to [nb, nk])
    # Col 5 = LDA, Col 6 = GW
    elda = data[:, 5].reshape((nb, nk))
    eqp = data[:, 6].reshape((nb, nk))

    # 4. Align VBM to 0 eV
    # We need to find which ROW of the array corresponds to band 'nv'
    try:
        # np.where finds the index where band_number == nv
        vbm_idx = np.where(bands_list == nv_bands)[0][0]
        cbm_idx = np.where(bands_list == (nv_bands + 1))[0][0]  # Next band is CBM

        # Align LDA VBM to 0
        vbm_val_lda = np.amax(elda[vbm_idx])
        elda -= vbm_val_lda
        gap_lda = np.amin(elda[cbm_idx]) - np.amax(elda[vbm_idx])

        # Align GW VBM to 0
        vbm_val_gw = np.amax(eqp[vbm_idx])
        eqp -= vbm_val_gw
        gap_gw = np.amin(eqp[cbm_idx]) - np.amax(eqp[vbm_idx])

    except IndexError:
        print(f"CRITICAL WARNING: VBM Band {nv_bands} not found in {filepath}.")
        print(f"    -> The file contains bands {bands_list[0]:.0f} to {bands_list[-1]:.0f}.")
        print(f"    -> Energies will NOT be aligned to 0 eV.")
        gap_lda, gap_gw = 0.0, 0.0

    return x_axis, eqp, elda, gap_gw, gap_lda


# ==========================================
# 3. PLOTTING LOGIC
# ==========================================

print(f"--- Plotting {len(datasets)} datasets (VBM Index={nv}) ---")

loaded_data = []
for ds in datasets:
    res = load_bands(ds['file'], nv)
    if res[0] is not None:
        loaded_data.append((ds, *res))

if not loaded_data:
    sys.exit("No data loaded.")

# Setup Figure
rc = matplotlib.rc
rc('figure', figsize=(9.0, 7.0))  # Slightly wider
rc('lines', linewidth=band_linewidth)
rc('font', size=16.0)
rc('axes', linewidth=1.5)
fig, ax = plt.subplots()

# Loop through datasets
for i, (ds, x, gw, lda, g_gw, g_lda) in enumerate(loaded_data):

    # Get color (use cycle if not defined in ds)
    color = ds.get('color', f"C{i}")
    label = ds['label']
    nb_bands = gw.shape[0]

    # Plot Bands
    for b in range(nb_bands):

        # --- LEGEND LOGIC ---
        # We only attach the label to the first band (b=0) to avoid duplication
        if b == 0:
            lbl_gw = f"{label} GW ($E_g$={g_gw:.2f})"
            lbl_lda = f"{label} LDA ($E_g$={g_lda:.2f})"
        else:
            lbl_gw = None
            lbl_lda = None

        # 1. Plot GW (Solid Line)
        ax.plot(x, gw[b], color=color, linestyle='-', label=lbl_gw, lw=band_linewidth)

        # 2. Plot LDA (Dotted/Dashed Line)
        if plot_lda:
            ax.plot(x, lda[b], color=color, linestyle=':', label=lbl_lda,
                    lw=band_linewidth, alpha=lda_alpha)

# --- AXIS & GRID ---
x_ref = loaded_data[0][1]

# K-Points X-ticks
if len(k_special_index) > 0:
    # Safety check: ensure indices exist in x_ref
    valid_mask = k_special_index < len(x_ref)
    k_locs = x_ref[k_special_index[valid_mask]]
    k_lbls = k_special_label[valid_mask]

    ax.set_xticks(k_locs)
    ax.set_xticklabels(k_lbls)

    # Vertical grid lines for K-points
    for val in k_locs:
        ax.axvline(val, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)

# Zero Energy Line
ax.axhline(0, color='black', linestyle='--', linewidth=0.5, alpha=0.5)

# Limits and Labels
ax.set_xlabel('Wavevector')
ax.set_ylabel('Energy (eV)')
ax.set_xlim(x_ref[0], x_ref[-1])
ax.set_ylim(emin, emax)

# Tick Formatting
ax.tick_params(direction='in', length=6, width=1.5, top=True, right=True)

# Legend placement (Upper Right, transparent box)
ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.02),
          ncol=len(datasets), frameon=False, fontsize='xx-small')

plt.tight_layout()
plt.savefig(output_filename, dpi=300)
print(f"Success! Saved plot to {output_filename}")