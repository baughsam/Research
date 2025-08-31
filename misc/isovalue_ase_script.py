from ase.io.cube import read_cube_data
import numpy as np
import sys #return isovalue that shows thres % of the charge density
def return_iso(file: str,thres: float,density: bool,single_prec: bool,num_acc: float):
    data, atoms = read_cube_data(file)
    if not density:
        data = data**2
    if single_prec:
        data = remove_below_precision(data,num_acc)
    #electron density are stored in voxels
    # we are calculating percent of density as a cumulative sum of a sorted array
    raveled = np.ravel(data)
    sorted_data = raveled[np.argsort(raveled)]
    percent = 100-np.cumsum(sorted_data)/np.sum(sorted_data)*100
    #get the last value
    iso_ind = np.where(percent>thres)[0][-1]
    return sorted_data[iso_ind]

#Gaussian prints out numbers below precision
#Using this or not may not make a difference
def remove_below_precision(array,prec):
    array[np.where(array<prec)]*=0
    return array
if __name__ == "__main__":
    #for debugging
    file = "cx_pent_3D_DEN.cube"
    thres = 40 #percent coverage desired
    density = True #gaussian files may not be density, VASP files usually are
    single_prec = True #you can check this in the cube file
    num_acc=np.finfo(np.float32).eps #for single precision    #for publication
    #filename, threshold, density or wf?
    #file = sys.argv[1]
    #thres = float(sys.argv[2])
    #density = boolean(sys.argv[3])
    print(return_iso(file,thres,density,single_prec,num_acc))