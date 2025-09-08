from ase.io import read, write
from ase.build.supercells import make_supercell
import numpy as np
from pymatgen.core.units import bohr_to_ang

from coord_transformations.primitive_to_supercell_1 import T11_1, T12_1

file_info =  read("bands_cx_pent.in")

M = [[8, 0, 0], [0, 8, 0], [0, 0, 4]] #supercell transformation matrix
supercell = make_supercell(file_info, M)


#cartesian to fractional coordinate transformation
# Supercell coordinates (cartesian)

T = [[6.266, 0.7203432, 0.587676], [0, 7.741558672, 3.358122], [0, 0, 14.1244]] #pentacene
#T = [[-2.69880378,    0.0000000000000000,    -2.69880378],[0,   2.69880378,   2.69880378],[2.69880378,   2.69880378,  0]]#Silicon
T_inv = np.linalg.inv(T)


vector_cart = np.array([4.7032866222716345,  3.0516558492139652,  0.1823525097028298])
vector_frac = np.array([4, 4, 2])
result_vector_frac = np.matmul(T_inv, vector_cart)
result_vector_cart = np.matmul(T, vector_frac)


print("\nResulting Vector (Cart):\n", result_vector_cart)
print("\nResulting Vector (Frac):\n", result_vector_frac)



#adding "hole" into supercell
#I'm trying to see if 4,4,2 will translate in supercell fractional coordinates or nah
hole = 'Ag'
hole_pos1 = (29.1207248,  37.68247869, 28.2488) #idk if this is in supercell coordinates #This is not supercell coordinates, these are real coordinates
#hole_pos2 = (26.64985518, 30.04484717, 29.06360005 )
#hole_pos3 = (28.54900136, 28.48860623, 31.87567007)
#hole_pos4 = (27.10161869, 29.86885381, 31.87567007)
supercell.append(hole)
supercell.positions[-1] = hole_pos1
#supercell.append(hole)
#supercell.positions[-1] = hole_pos2
#supercell.append(hole)
#supercell.positions[-1] = hole_pos3
#supercell.append(hole)
#supercell.positions[-1] = hole_pos4
write('supercell_hole.cif', supercell, format='cif')