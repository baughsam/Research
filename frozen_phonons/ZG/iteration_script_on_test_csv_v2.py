#this script takes the csv files with positions and eigenvectors,
# runs each of the eigenvectors through the temperature dependence funciton (from ZG)
# adds the dep_dependent_eigenfunctions to the positions
# puts all the information into another csv file

import pandas as pd

filename = 'test_csv'
nat = 3 #72
numc = 1 # number of carbon atoms
numh = 2 # number of hydrogen atoms
#carbons need to come right after hydrogens or vice versa #THEY CANNOT MIX OF CODE WILL NOT WORK



def ex_temp_disp_funct(eigval,mass_k):
    ex = eigval * mass_k
    return ex


df = pd.read_csv(filename+".csv")
df_list = df.values.tolist() #puts .csv file into list
col_names = ['X_T_dep', 'Y_T_dep', 'Z_T_dep', 'Eigenvectors_(dx)_T_dep', 'Eigenvectors_(dy)_T_dep', 'Eigenvectors_(dz)_T_dep', 'X', 'Y', 'Z', 'Eigenvectors_(dx)', 'Eigenvectors_(dy)', 'Eigenvectors_(dz)']
pos_data = []

for k in range(nat):
    unmod_pos = df_list[k][:3]
    unmod_eig = df_list[k][3:]
    atom_pos_data = []
    mod_eig_list = []
    temp_dep_pos_list = []
    if k in range(numc):
        for j in range(3):
            mod_eig = ex_temp_disp_funct(unmod_eig[j], 10)
            pos_temp_dep = unmod_pos[j] + mod_eig
            mod_eig_list.append(mod_eig)
            temp_dep_pos_list.append(pos_temp_dep)
        pos_data_list = temp_dep_pos_list + mod_eig_list + unmod_pos + unmod_eig
        pos_data.append(pos_data_list)
    if k in range (numc, numc+numh):
        for j in range(3):
            mod_eig = ex_temp_disp_funct(unmod_eig[j], 100)
            pos_temp_dep = unmod_pos[j] + mod_eig
            mod_eig_list.append(mod_eig)
            temp_dep_pos_list.append(pos_temp_dep)
        pos_data_list = temp_dep_pos_list + mod_eig_list + unmod_pos + unmod_eig
        pos_data.append(pos_data_list)

df_1 = pd.DataFrame(pos_data, columns=col_names)
file_path = filename + "_mod.csv"
# Save the DataFrame to a CSV file
df_1.to_csv(file_path, index=False)
print(pos_data)

