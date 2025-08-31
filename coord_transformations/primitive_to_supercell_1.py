from ase.io import read, write
from ase.build.supercells import make_supercell
import numpy as np


file_info =  read("bands_cx_pent.in")

M = [[8, 0, 0], [0, 8, 0], [0, 0, 4]] #supercell transformation matrix
supercell = make_supercell(file_info, M)

#cartesian to fractional coordinate transformation
#Note (07-02) the lattice parameters need to be in fractional coordinates, which is why this isn't working
#Lattice parameters (Angstrom)
a = 50.128
b = 62.20
c = 58.12
#angles (degrees)
alpha = 76.475
beta = 87.682
gamma = 84.684
alpha_rad = np.radians(alpha)
beta_rad = np.radians(beta)
gamma_rad = np.radians(gamma)

#row 1
T11_1 = 1/np.sin(beta_rad)
T11_2 = a * (1-((1/np.tan(alpha_rad))*(1/np.tan(beta)) * 1/np.sin(alpha_rad) * 1/np.sin(beta_rad) * np.cos(gamma_rad))**2)**0.5

T11 = T11_1/T11_2
T21 = 0
T31 = 0

#row2
T12_1 = (((1/np.sin(alpha_rad))**2) * 1/np.sin(beta_rad)) * (np.cos(alpha_rad)*np.cos(beta_rad)-np.cos(gamma_rad))
T12_2 = b * ((1-((1/np.tan(alpha_rad))*(1/np.tan(beta_rad)) * 1/np.sin(alpha_rad) * 1/np.sin(beta_rad) * np.cos(gamma_rad))**2)**0.5)

T12 = T12_1 / T12_2
T22 = (1/np.sin(alpha_rad)) / b
T32 = 0
print(T12)
#row3
T13_1 = (1/np.sin(alpha_rad)) * ( 1/np.tan(alpha_rad) * 1/np.sin(beta_rad) * np.cos(gamma_rad) - 1/np.sin(alpha_rad) * 1/np.tan(beta_rad) )
T13_2 = c * (1-((1/np.tan(alpha_rad))*(1/np.tan(beta_rad)) * 1/np.sin(alpha_rad) * 1/np.sin(beta_rad) * np.cos(gamma_rad))**2)**0.5

T13 = T13_1/T13_2
T23 = - (1/np.tan(alpha_rad))/(c)
T33 = 1/c

print(T13)
T      = np.array([[T11, T21, T31],
                   [T12, T22, T32],
                   [T13, T23, T33]])

vector = np.array([29.12072,  37.68248, 28.24879])

result_vector = np.matmul(T, vector)

print("Matrix:\n", T)
print("\nVector:", vector)
print("\nResulting Vector (using matmul):\n", result_vector)





#adding "hole" into supercell
#I'm trying to see if 4,4,2 will translate in supercell fractional coordinates or nah
hole = 'Ag'
hole_pos = (4, 4, 2) #idk if this is in supercell coordinates #This is not supercell coordinates, these are real coordinates

#hole at 4,4,2 translates to
#(x,y,z):  29.12072  37.68248  28.24879
#in supercell fractional units




#supercell.append(hole)
#supercell.positions[-1] = hole_pos
#print(supercell.positions)
#write('supercell_hole.cif', supercell, format='cif')