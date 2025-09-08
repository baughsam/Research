import math
import numpy as np
from ase import Atoms
import pandas as pd
from ase.io.vasp import write_vasp, read_vasp

#tau_ZG_displacement script#
#bose-einstein occupation as given in https://doi-org.ezproxy.bu.edu/10.1103/
def bose_einstein_occupation(freq, temperature):
    boltzmann_constant = 1.38065e-23 #Joules per Kelvin
    reduced_planck = 1.05457e-34 # Joules times second
    ang_freq = freq * 2 * math.pi

    exp1 = np.exp((reduced_planck * ang_freq ) / (boltzmann_constant * temperature))
    n = (exp1 - 1) ** -1
    return n

#tau displacment as given in https://doi-org.ezproxy.bu.edu/10.1103/
def tau_displacement(atomic_mass, freq, temperature):
    #atomic_mass units ~ kg
    #freq units        ~ 1 / second

    reduced_planck = 1.05457e-34  # Joules times second
    bose_ein = bose_einstein_occupation(freq, temperature)
    ang_freq = freq * 2 * math.pi

    m1 = 2*bose_ein + 1
    m2 = reduced_planck / (2 * atomic_mass * ang_freq)
    disp = (m1 * m2) ** (1/2) #in meters
    disp_ang = disp * 1e10  # conversion to angstroms

    return disp_ang






single_phon_freq = 2.703425e12 #value from 91cm-1 mode for -cx ~ 1 / second
mass_h = 1.67e-27 #~ mass of hydrogen in kg
mass_c = 1.99e-26 #~ mass of carbon in kg
Temp = 300 #~ Kelvin
#####################################
nat = 72
numc = 44 # number of carbon atoms #44
numh = 28 # number of hydrogen atoms #28
#carbons need to come right after hydrogens or vice versa #THEY CANNOT MIX OF CODE WILL NOT WORK
#####################################
perc = .9 #percentage (in decimals) of the zg displacement applied to the original displacement

filename = '../Complete_ZG_code_pent/90.176552_cm-1_posvec+eigvec'
#filename = '../90.176552_cm-1_posvec+eigvec'

#Iteration Script#
df = pd.read_csv(filename+".csv")
df_list = df.values.tolist() #puts .csv file into list
col_names = ['X_T_dep', 'Y_T_dep', 'Z_T_dep', 'Eigenvectors_(dx)_T_dep', 'Eigenvectors_(dy)_T_dep', 'Eigenvectors_(dz)_T_dep', 'X', 'Y', 'Z', 'Eigenvectors_(dx)', 'Eigenvectors_(dy)', 'Eigenvectors_(dz)']
pos_data = []
print(range(numc))
print(range(numc, numh+numc))
for k in range(nat):
    unmod_pos = df_list[k][:3]
    unmod_eig = df_list[k][3:]
    atom_pos_data = []
    mod_eig_list = []
    temp_dep_pos_list = []
    if k in range(numc):
        for j in range(3):
            mod_eig = tau_displacement(mass_c, single_phon_freq, Temp) * unmod_eig[j]
            pos_temp_dep = unmod_pos[j] + (mod_eig * perc) #I believe tat multipying the mod_eig by a percentage value (e.g. 10% means we multiply by .1) will do the trick
            mod_eig_list.append(mod_eig)
            temp_dep_pos_list.append(pos_temp_dep)
        pos_data_list = temp_dep_pos_list + mod_eig_list + unmod_pos + unmod_eig
        pos_data.append(pos_data_list)
    if k in range (numc, numc+numh):
        for j in range(3):
            mod_eig = tau_displacement(mass_h, single_phon_freq, Temp) * unmod_eig[j]
            pos_temp_dep = unmod_pos[j] + (mod_eig * perc)
            mod_eig_list.append(mod_eig)
            temp_dep_pos_list.append(pos_temp_dep)
        pos_data_list = temp_dep_pos_list + mod_eig_list + unmod_pos + unmod_eig
        pos_data.append(pos_data_list)
#print(pos_data)
df_1 = pd.DataFrame(pos_data, columns=col_names)
filename_mod = filename + "_mod_"+str(perc*100)+"_perc.csv"
# Save the DataFrame to a CSV file
df_1.to_csv(filename_mod, index=False)


#csv_to_POSCAR script#
#extracting information from csv file
mod_csv_df = pd.read_csv(filename_mod, usecols=['X_T_dep', 'Y_T_dep', 'Z_T_dep']) #read in
pos_array = mod_csv_df.values
#print(pos_array)
#making atom object
phon_atoms = Atoms('C44H28')
#cell is manually put in for now, but we should grab it from the CONTCAR
cell_array = [[6.2660000000000000,    0.0000000000000000,    0.0000000000000000],[0.7203431964649767,   7.7415586724707204,   0.0000000000000000],[0.5876759734010517,   3.3581219057453895,  14.1243957115495764]]
phon_atoms.set_positions(pos_array) # setting positions of atoms
phon_atoms.set_cell(cell_array)


write_vasp(file=filename+"_mod_"+str(perc*100)+"_perc.vasp", atoms=phon_atoms, direct = False)
