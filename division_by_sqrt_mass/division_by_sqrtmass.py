import numpy as np
import math

outcar = open("OUTCAR.txt", "r")
nat = 72 #number of atoms in system
numc = 44 #number of carbons in system
numh = 28 #number of hydrogens

c_mass = 12.011 #mass of carbon
h_mass = 1      #mass of hydrogen

sqrt_c_mass = math.sqrt(c_mass) #sqrt mass of carbon
sqrt_h_mass = math.sqrt(h_mass) #sqrt mass of hydrogen

f1 = open("sqrtmass.txt", "w")
f1.write(" Eigenvectors after division by SQRT(mass)\n\n")

def truncate(f, n):
    '''Truncates/pads a float f to n decimal places without rounding'''
    s = '{}'.format(f)
    if 'e' in s or 'E' in s:
        return '{0:.{1}f}'.format(f, n)
    i, p, d = s.partition('.')
    return '.'.join([i, (d+'0'*n)[:n]])


while True:
    line = outcar.readline()
    if not line:
        break
    if "Eigenvectors and eigenvalues of the dynamical matrix" in line:
        f1.write(line)
        f1.write(outcar.readline()) # ----------------------------------------------------
        f1.write(outcar.readline()) # empty line
        for i in range(nat*3):
            f1.write(outcar.readline()) #empty line
            f1.write(outcar.readline()) #frequency line
            f1.write(outcar.readline()) #empty line
            for i in range(numc): #Carbon Division
                s = outcar.readline()
                whole_list = s.split()
                pos_list = whole_list[:3]
                eig_list = whole_list[3:]
                #
                eig_list = [float(item) for item in eig_list] #turns list of strings to float for the np array
                eig_array = np.array(eig_list)
                sqrtmass_eig_array = np.divide(eig_array,sqrt_c_mass)
                sqrtmass_eig_list = np.array(sqrtmass_eig_array).tolist() #turns array into list
                #
                for x in range(3):
                    sqrtmass_eig_list[x] = truncate(sqrtmass_eig_list[x], 6)
                #
                sqrtmass_eig_list = [str(item) for item in sqrtmass_eig_list] #turns list of floats to string
                mod_whole_list = pos_list + sqrtmass_eig_list
                eig_string = " ".join(mod_whole_list)
                f1.write(eig_string+"\n")
                #print(eig_string)
            for i in range(numh): #Hydrogen division
                s = outcar.readline()
                whole_list = s.split()
                pos_list = whole_list[:3]
                eig_list = whole_list[3:]

                eig_list = [float(item) for item in eig_list] #turns list of strings to float for the np array
                eig_array = np.array(eig_list)
                sqrtmass_eig_array = np.divide(eig_array,sqrt_h_mass)
                sqrtmass_eig_list = np.array(sqrtmass_eig_array).tolist()

                for x in range(3):
                    sqrtmass_eig_list[x] = truncate(sqrtmass_eig_list[x], 6)

                sqrtmass_eig_list = [str(item) for item in sqrtmass_eig_list] #turns list of floats to string
                mod_whole_list = pos_list + sqrtmass_eig_list
                eig_string = " ".join(mod_whole_list)
                f1.write(eig_string+"\n")
f1.close()

#NOTES
#This can be simplified, but I was mostly trying to get this to work

#There is a deviation from the VASP format in the xyz and eigenvectors. I've yet to try, but I do not think this will affect raman-sc's reading of the information.
#Note(05-30): Format does not affect reading of information


