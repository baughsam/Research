from ase import Atoms
import pandas as pd
from ase.io.vasp import write_vasp, read_vasp

#extracting information from csv file
mod_csv_df = pd.read_csv('test_csv_mod.csv', usecols=['X_T_dep', 'Y_T_dep', 'Z_T_dep']) #read in
pos_array = mod_csv_df.values

#making atom object
phon_atoms = Atoms('C3')
#cell is manually put in for now, but we should grab it from the CONTCAR
cell_array = [[6.2660000000000000,    0.0000000000000000,    0.0000000000000000],[0.7203431964649767,   7.7415586724707204,   0.0000000000000000],[0.5876759734010517,   3.3581219057453895,  14.1243957115495764]]
phon_atoms.set_positions(pos_array) # setting positions of atoms
phon_atoms.set_cell(cell_array)


write_vasp(file="POSCAR_T_dep.vasp", atoms=phon_atoms, direct = False)