from pylab import *
import sys
import re
import pandas as pd
import numpy as np

scaling_factor = 5
thickness = 0.25
R = 150 #0
G = 0 #255
B = 255 #0
color = f'{R:3d} {G:3d} {B:3d}'
Cart_true = "False"  ## False = convert to fractional, True = convert to cartesian

def MAT_m_VEC(m, v):
    p = [ 0.0 for i in range(len(v)) ]
    for i in range(len(m)):
        assert len(v) == len(m[i]), 'Length of the matrix row is not equal to the length of the vector'
        p[i] = sum( [ m[i][j]*v[j] for j in range(len(v)) ] )
    return p


def T(m):
    p = [[ m[i][j] for i in range(len( m[j] )) ] for j in range(len( m )) ]
    return p

def Coordinate_convert(b, coord, Cart_true): 

    A = [b[0][0], b[0][1], b[0][2]]
    B = [b[1][0], b[1][1], b[1][2]]
    C = [b[2][0], b[2][1], b[2][2]]
    # print(A)
    # print(B)
    # print(C)
    # Build the matrix of lattice vectors stored column-wise
    # and get its inverse
    M = np.vstack([A, B, C]).T
    M_inv = np.linalg.inv(M)
    # A set of coordinates stored row-wise (numpy array)
    Y = coord
    # Compute the cartesian coordinates
    if Cart_true == "True":
        X = np.matmul(M, Y.T).T
        # print("cartesian coodinates: ")
        # print(X)
    # Perform the inverse operation to get fractional coordinates
    if Cart_true == "False": 
        X = np.matmul(M_inv, Y.T).T
        print("fractional coodinates: ")
        print(X)
    return X

def calc_freq_displacement_for_one_lattice(b, EV_array, Cart_True): 
    #subroutine to take the eignenvector displacements, converte to fractional coord and pull out individual contributions
    coord = np.array(EV_array)
    # print(coord)
    Cart_true = Cart_True
    XYZ = Coordinate_convert(b, coord, Cart_true)
    A_proj = XYZ[0]
    B_proj = XYZ[1]
    C_proj = XYZ[2]

    return A_proj, B_proj, C_proj


def parse_poscar(poscar):
    # modified subroutine from phonopy 1.8.3 (New BSD license)
    poscar.seek(0) # just in case
    lines = poscar.readlines()

    scale = float(lines[1])
    if scale < 0.0:
        print("[parse_poscar]: ERROR negative scale not implemented.")
        sys.exit(1)

    b = []
    for i in range(2, 5):
        b.append([float(x)*scale for x in lines[i].split()[:3]])

    vol = b[0][0]*b[1][1]*b[2][2] + b[1][0]*b[2][1]*b[0][2] + b[2][0]*b[0][1]*b[1][2] - \
          b[0][2]*b[1][1]*b[2][0] - b[2][1]*b[1][2]*b[0][0] - b[2][2]*b[0][1]*b[1][0]

    try:
        num_atoms = [int(x) for x in lines[5].split()]
        line_at = 6
    except ValueError:
        symbols = [x for x in lines[5].split()]
        num_atoms = [int(x) for x in lines[6].split()]
        line_at = 7
    # print(num_atoms)
    nat = sum(num_atoms)
    natC = num_atoms[0]

    if lines[line_at][0].lower() == 's':
        line_at += 1

    if (lines[line_at][0].lower() == 'c' or lines[line_at][0].lower() == 'k'):
        is_scaled = False
    else:
        is_scaled = True

    line_at += 1
    positions = []
    for i in range(line_at, line_at + nat):
        pos = [float(x) for x in lines[i].split()[:3]]
        if is_scaled:
            pos = MAT_m_VEC(T(b), pos)
        positions.append(pos)
    poscar_header = ''.join(lines[1:line_at-1]) # will add title and 'Cartesian' later

    return nat, natC, vol, b, positions, poscar_header

def parseModes(outcar, nat, natC, vesta_front, vesta_end, scaling_factor, b, Cart_True):

    eigvals = [ 0.0 for i in range(nat*3) ]
    EVAL = []
    eigvecs = [ 0.0 for i in range(nat*3) ]
    norms   = [ 0.0 for i in range(nat*3) ]
    EVECx = []
    EVECy = []
    EVECz = []
    Aproj = []
    Bproj = []
    Cproj = []
    abs_Aproj = []
    abs_Bproj = []
    abs_Cproj = []
    NORMS = []

    outcar.seek(0) # just in case
    while True:
        line = outcar.readline()
        if not line:
            break
        # if "LATTYP: Found" in line: 
        #     outcar.readline() # ----------------------------------------------------
        #     ALAT = re.search(r'^\s*(\d+).+?([\.\d]+) cm-1', outcar.readline())
        if "Eigenvectors and eigenvalues of the dynamical matrix" in line:
            outcar.readline() # ----------------------------------------------------
            outcar.readline() # empty line
            print("Mode    Freq (cm-1)")
            for i in range(nat*3):
                outcar.readline() # empty line
                p = re.search(r'^\s*(\d+).+?([\.\d]+) cm-1', outcar.readline())
                eigvals[i] = float(p.group(2))

                outcar.readline() # X         Y         Z           dx          dy          dz
                eigvec = []

                for j in range(nat):
                    tmp = outcar.readline().split()
                    eigvec.append([ float(tmp[x]) for x in range(3,6) ])
                    if j < natC: 
                        EVAL +=[float(p.group(2))]
                        EVECx +=[float(tmp[3])]
                        EVECy +=[float(tmp[4])]
                        EVECz +=[float(tmp[5])]
                        NORMS +=[sqrt( sum( (abs(float(tmp[3])))**2 + (abs(float(tmp[4])))**2 + (abs(float(tmp[5])))**2))]
                        eigvec_list = [float(tmp[3]), float(tmp[4]), float(tmp[5])]
                        # if i == 0:
                        #     # print("new routine")
                        #     print(eigvec_list)
                        #     AP, BP, CP = calc_freq_displacement_for_one_lattice(b, eigvec_list, Cart_True)
                        #     print(AP, BP, CP)
                        AP, BP, CP = calc_freq_displacement_for_one_lattice(b, eigvec_list, Cart_True)
                        Aproj +=[AP]
                        Bproj +=[BP]
                        Cproj +=[CP]
                        # print(AP, BP, CP)
                        abs_Aproj +=[abs(AP)]
                        abs_Bproj +=[abs(BP)]
                        abs_Cproj +=[abs(CP)]
                    
                eigvecs[i] = eigvec
                norms[i] = sqrt( sum( [abs(x)**2 for sublist in eigvec for x in sublist] ) )
                writeVestaMode(i, eigvals[i], eigvecs[i], vesta_front, vesta_end, natC, scaling_factor)
                print("%4d      %6.2f" %(i+1, eigvals[i]))

        if "Eigenvectors after division by SQRT(mass)" in line:
            break
        if "ELASTIC MODULI CONTR FROM IONIC RELAXATION (kBar)" in line:
            break
    #NORM = [float(t) for t in NORMS]
    df_eig = {"Eigenfrequency (cm-1)": EVAL, "Eigenvector (dx)": EVECx, "Eigenvector (dy)": EVECy, "Eigenvector (dz)": EVECz, "Eigenvector mag": NORMS,"A projection":Aproj, "abs A projection": abs_Aproj, "B projection": Bproj, "abs B projection": abs_Bproj,  "C projection": Cproj, "abs C projection": abs_Bproj}
    df_eig = pd.DataFrame(df_eig)
    df_eig.to_csv("Eignfrequency_info.csv", index = False)
    grouped = df_eig.groupby(["Eigenfrequency (cm-1)"])
    for group_name, group_data in grouped:
        group_n = str(group_name)
        gn = re.findall(r"[-+]?(?:\d*\.*\d+)", group_n)
        group_data.to_csv(gn[0]+'_Frequency_per_atom_data.csv', index=False)
    grouped2 = df_eig.groupby(["Eigenfrequency (cm-1)"]).sum()
    grouped2.to_csv('Summed_over_all-atoms_Eignfrequency_info.csv')

    return eigvals, eigvecs, norms


def writeVestaMode(i, eigval, eigvec, vesta_front, vesta_end, natC, scaling_factor):
    modef = open("mode_%.2f.vesta"%eigval, 'w')

    modef.write(vesta_front)

    sf = scaling_factor
    towrite = "VECTR\n"
    for i in range(1,1+natC):
        towrite += "%4d%9.5f%9.5f%9.5f\n"%(i,eigvec[i-1][0]*sf,eigvec[i-1][1]*sf,eigvec[i-1][2]*sf)
        towrite += "%5d  0   0    0    0\n 0 0 0 0 0\n"%i
    towrite += " 0 0 0 0 0\n"
    towrite += "VECTT\n"
    for i in range(1,1+nat):
        towrite += f'{i:4d} {thickness:2.3f} {color} 0\n' # 0 for non-penetrating arrow, 1 for penetrating arrow

    if i==0:
        print(towrite)
        
    modef.write(towrite)

    return 0


def openVestaOutcarPoscar():
    if len(sys.argv) == 1:
        try:
            vesta  = open('poscar.vesta','r')
        except:
            print("Cannot find poscar.vesta in current directory")
            print("Usage:\n\tpython modes_to_vesta.py <vesta-filename.vesta>")
            sys.exit(0)
    elif len(sys.argv) == 2:
        try:
            print("Opening ", sys.argv[1])
            vesta = open(sys.argv[1],'r')
        except:
            print("Cannot find file ", sys.argv[1])
            sys.exit(0)
    else:
        print("Cannot parse >1 command-line argument")
        sys.exit(0)

    try:
        outcar = open('OUTCAR', 'r')
    except:
        print("Cannot find OUTCAR in current directory")
        sys.exit(0)

    try:
        poscar = open('POSCAR', 'r')
    except:
        print("Cannot find POSCAR in current directory")
        sys.exit(0)

    return vesta, outcar, poscar


def getVestaFrontEnd(vesta):

    vfile = vesta.read()
    vesta_front = vfile.split("VECTR")[0]
    vesta_end   = vfile.split("VECTT\n 0 0 0 0 0")[1]

    return vesta_front, vesta_end

if __name__ == '__main__':

    vesta, outcar, poscar = openVestaOutcarPoscar()
    vesta_front, vesta_end = getVestaFrontEnd(vesta)
    nat, natC, vol, b, positions, poscar_header = parse_poscar(poscar)

    print("# atoms   vol of unit cell (Ang^3)   # modes")
    print("  %d      %4.2f       %d" %(nat,vol,nat*3))

    parseModes(outcar, nat, natC, vesta_front, vesta_end, scaling_factor, b, Cart_true)

    # coord = np.array([4.696232, 3.164971, 0.216165])
    # coord = np.array([[0.000345, 0.066682, 0.124008], 
    #           [0.000156, 0.066388, 0.124334],
    #           [0.001843, 0.070655, 0.136102]]
    #          )
    # # #coord = np.array([[0.7018071412581293, 0.4021899471312835, 0.0153043539261617], 
    # # #          [0.2981928587418707, 0.5978100528687165, 0.9846956460738383]]
    # # #        )
    # Coordinate_convert(b, coord, Cart_true)

