from ase.io.vasp import write_vasp, read_vasp
from ase.geometry import wrap_positions
from pandas import read_csv
import numpy as np

#scaling the amount of displacement we want
eig_frac = 1

#getting carbons from CONTCAR file
b = read_vasp("CONTCAR_91")
pos = b.get_positions()
pos_c = pos[:44]
pos_h = pos[44:]
Cell = b.get_cell()


#getting eigenvalue positions from csv file
usecols = ["Eigenvector (dx)", "Eigenvector (dy)", "Eigenvector (dz)"]
csv = read_csv("90.176552_Frequency_per_atom_data.csv", usecols=usecols)
eigval_array = eig_frac * csv.to_numpy()


#displacing CONTCAR positions by the eigenvalue positions
frozen_pos = np.add(pos_c,eigval_array)


#putting all atomic positions back together
pos_all = np.concatenate([frozen_pos,pos_h])

#Double checking that all atoms are inside of the unit cell
new_atoms = wrap_positions(pos_all, Cell)
#setting positions in the Atom object
b.set_positions(new_atoms, apply_constraint=False)



#phon_atoms = Atoms(symbols="C44H28", pbc=True, cell=Cell, positions=b2)
#write_vasp(file="POSCAR.vasp", atoms=b)
