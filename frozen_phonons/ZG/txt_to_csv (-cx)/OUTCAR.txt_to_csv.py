#test code
#converting OUTCAR from phonon calculations into a csv file with the labels
# X, Y, Z, Eigenvector (dx), Eigenvector (dy), Eigenvector (dz)

import re
import pandas as pd

outcar = open("OUTCAR.txt", "r")
nat = 72 #number of atoms in pentacene

freq_list = []
col_names = ['X', 'Y', 'Z', 'Eigenvectors_(dx)', 'Eigenvectors_(dy)', 'Eigenvectors_(dz)']
while True:
    line = outcar.readline()
    if not line:
        break
    if "Eigenvectors and eigenvalues of the dynamical matrix" in line:
        outcar.readline() # ----------------------------------------------------
        outcar.readline() # empty line
        for i in range(nat*3):
            pos_data =[]
            outcar.readline() # empty line
            FREQ = re.search(r'^\s*(\d+).+?([\.\d]+) cm-1', outcar.readline()) # searches line for frequency in cm-1
            freq_list.append(FREQ.group(2)) # this is a float value (i think)
            outcar.readline() #             X         Y         Z           dx          dy          dz
            for j in range(nat): #72 atoms
                s = outcar.readline()
                whole_list = s.split()
                whole_list = [float(item) for item in whole_list]  # turns list of strings to float
                pos_data.append(whole_list)
            df = pd.DataFrame(pos_data, columns=col_names) # Create a DataFrame from the list
            file_path = freq_list[i] + "_cm-1_posvec+eigvec.csv"
            # Save the DataFrame to a CSV file
            df.to_csv(file_path, index=False)  # index=False prevents writing the DataFrame index as a column


