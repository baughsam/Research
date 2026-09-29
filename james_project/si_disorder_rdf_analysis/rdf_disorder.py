import numpy as np
import matplotlib.pyplot as plt
from ase.io import read
from ase.geometry.analysis import Analysis

# Load the perfectly base cell
base_atoms = read('si_54.data', format='lammps-data', style='atomic')

bond_length = 2.35 # Si-Si equilibrium bond length is ~2.35 A

# input initial and final percentages
perc_i = 0
perc_f = 0.08
num_rdf = 25

#rdf inputs
rmax = 4.6 #angstrom
nbins = 200

# Test 11 variations from 0% to 25% displacement
percentages = np.linspace(perc_i, perc_f, num_rdf)

plt.figure(figsize=(10, 8))

for i, p in enumerate(percentages):
    # Clone structure to isolate each perturbation
    atoms = base_atoms.copy()

    # Rattle applies a Gaussian distribution. stdev is in Angstroms.
    stdev = bond_length * p
    atoms.rattle(stdev=stdev, seed=42)

    # Calculate RDF up to 6.0 Angstroms
    ana = Analysis(atoms)
    # Bin edges and centers
    edges = np.linspace(0.0, rmax, nbins + 1)
    r_distance = 0.5 * (edges[:-1] + edges[1:])
    # get_rdf returns a list of arrays; [0] grabs the single structure's RDF
    rdf = ana.get_rdf(rmax=rmax, nbins=nbins)[0]

    # Offset each plot vertically by i * 2 for clean visualization
    offset = i * 2
    plt.plot(r_distance, rdf + offset, label=f'{p * 100:.1f}% (stdev: {stdev:.2f} Å)')

plt.xlabel('Distance (Å)', fontsize=12)
plt.ylabel('Radial Distribution Function (offset)', fontsize=12)
plt.title('Structural Washout via Gaussian Perturbation', fontsize=14)

plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig(fname=f"si_rdf_{perc_i}-{perc_f}_perc.png", dpi=300)
plt.show()