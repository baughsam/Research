#Takes data from eqp1.dat (SIGMA)
# Gives:
# Indirect Gap
# Direct Gap
# Lowest Direct Gap

#Modifications for Specific System:
# Valence Band Minimum (vbm_band)
# VBM for print statements



#!/usr/bin/env python3
import sys

def parse_bands(filename, energy_col=3):
    """
    Parse a band‐structure file with blocks of:
      kx ky kz    n_bands
       spin  band_index  col3  col4
       ...
    Returns:
      kpoints: list of (kx,ky,kz)
      band_data: dict mapping kpoint→{band_index: energy}
    """
    kpoints = []
    band_data = {}
    with open(filename) as f:
        while True:
            header = f.readline()
            if not header:
                break
            parts = header.split()
            if len(parts)==4 and '.' in parts[0]:
                kx, ky, kz = map(float, parts[:3])
                n_bands = int(parts[3])
                kp = (kx, ky, kz)
                kpoints.append(kp)
                band_data[kp] = {}
                for _ in range(n_bands):
                    line = f.readline().split()
                    band_idx = int(line[1])
                    energy   = float(line[energy_col])
                    band_data[kp][band_idx] = energy
    return kpoints, band_data

def compute_gaps(kpoints, band_data, vbm_band=102):
    # Build VBM energies and lists of (band, energy) for conduction
    vb = {kp: band_data[kp][vbm_band] for kp in kpoints}
    cb_map = {
        kp: [(b,e) for b,e in band_data[kp].items() if b > vbm_band]
        for kp in kpoints
    }

    # --- Indirect gap ---
    # VBM: highest among vb[kp]
    vbm_kp, VBM_global = max(vb.items(), key=lambda x: x[1])
    # CBM: lowest among all cb_map entries
    all_cb = [
        (kp, b, e)
        for kp, lst in cb_map.items()
        for b,e in lst
    ]
    cbm_kp, cbm_band, CBM_global = min(all_cb, key=lambda x: x[2])
    indirect_gap = CBM_global - VBM_global

    # --- Direct gap @ Γ ---
    gamma = (0.0, 0.0, 0.0)
    VBM_G = vb[gamma]
    cbmG_band, CBM_G = min(cb_map[gamma], key=lambda be: be[1])
    direct_gap_gamma = CBM_G - VBM_G

    # --- Lowest direct gap anywhere ---
    direct_list = []
    for kp in kpoints:
        v = vb[kp]
        b_min, e_min = min(cb_map[kp], key=lambda be: be[1])
        direct_list.append((kp, b_min, e_min - v))
    dir_kp, dir_band, lowest_direct = min(direct_list, key=lambda x: x[2])

    return {
        "indirect_gap":        indirect_gap,
        "vbm_kpoint":          vbm_kp,
        "vbm_energy":          VBM_global,
        "cbm_kpoint":          cbm_kp,
        "cbm_band":            cbm_band,
        "cbm_energy":          CBM_global,
        "direct_gamma_gap":    direct_gap_gamma,
        "direct_gamma_cbm_band": cbmG_band,
        "lowest_direct_gap":   lowest_direct,
        "lowest_direct_kpoint": dir_kp,
        "lowest_direct_band":  dir_band
    }

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} bands_file")
        sys.exit(1)

    fname = sys.argv[1]
    kpts, data = parse_bands(fname, energy_col=3)
    res = compute_gaps(kpts, data)

    print(f"Indirect gap:        {res['indirect_gap']:.6f} eV")
    print(f"  VBM at band 102     → k = {res['vbm_kpoint']}  (E = {res['vbm_energy']:.6f} eV)")
    print(f"  CBM at band {res['cbm_band']}     → k = {res['cbm_kpoint']}  (E = {res['cbm_energy']:.6f} eV)\n")

    print(f"Direct gap @ Γ:      {res['direct_gamma_gap']:.6f} eV")
    print(f"  VBM at band 102     → k = (0.0, 0.0, 0.0)")
    print(f"  CBM at band {res['direct_gamma_cbm_band']}     → k = (0.0, 0.0, 0.0)\n")

    print(f"Lowest direct gap:   {res['lowest_direct_gap']:.6f} eV")
    print(f"  VBM at band 102       → k = {res['lowest_direct_kpoint']}")
    print(f"  CBM at band {res['lowest_direct_band']}     → k = {res['lowest_direct_kpoint']}")
