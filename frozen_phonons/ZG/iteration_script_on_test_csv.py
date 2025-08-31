import pandas as pd

filename = 'test_csv.csv'
nat = 3 #72
numc = 1 # number of carbon atoms
numh = 2 # number of hydrogen atoms
#carbons need to come right after hydrogens or vice versa #THEY CANNOT MIX OF CODE WILL NOT WORK



def ex_temp_disp_funct(eigval,mass_k):
    ex = eigval * mass_k
    return ex


df = pd.read_csv(filename)
df_list = df.values.tolist() #puts .csv file into list
col_names = ['X_temp', 'Y_temp', 'Z_temp', 'Eigenvectors_(dx)_temp', 'Eigenvectors_(dy)_temp', 'Eigenvectors_(dz)_temp', 'X', 'Y', 'Z', 'Eigenvectors_(dx)', 'Eigenvectors_(dy)', 'Eigenvectors_(dz)']
pos_data = []
for i in range(numc):
    atom_pos_data = []
    unmod_pos = df_list[i][:3]
    unmod_eig = df_list[i][3:]
    mod_eig_list = []
    temp_dep_pos_list = []
    for j in range(3):
        mod_eig = ex_temp_disp_funct(unmod_eig[j],10)
        pos_temp_dep = unmod_pos[j] + mod_eig
        mod_eig_list.append(mod_eig)
        temp_dep_pos_list.append(pos_temp_dep)
    pos_data_list = temp_dep_pos_list + mod_eig_list + unmod_pos + unmod_eig
    pos_data.append(pos_data_list)
for i in range(numh):
    atom_pos_data = []
    unmod_pos = df_list[i][:3]
    unmod_eig = df_list[i][3:]
    mod_eig_list = []
    temp_dep_pos_list = []
    for j in range(3):
        mod_eig = ex_temp_disp_funct(unmod_eig[j],10)
        pos_temp_dep = unmod_pos[j] + mod_eig
        mod_eig_list.append(mod_eig)
        temp_dep_pos_list.append(pos_temp_dep)
    pos_data_list = temp_dep_pos_list + mod_eig_list + unmod_pos + unmod_eig
    pos_data.append(pos_data_list)

print(pos_data)
