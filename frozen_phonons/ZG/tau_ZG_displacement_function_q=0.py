import numpy as np

N_p = 1 # Number of unit cells in supercell
single_phon_freq = 2.703425e12 #value from 91cm-1 mode for -cx ~ 1 / second
M_h = 1.67e-27 #~ mass of hydrogen in kg
M_c = 1.99e-26 #~ mass of carbon in kg
Temp = 300



#bose-einstein occupation as given in https://doi-org.ezproxy.bu.edu/10.1103/
def bose_einstein_occupation(freq, temperature):
    boltzmann_constant = 1.38065e-23 #Joules per Kelvin
    reduced_planck = 1.05457e-34 # Joules times second
    exp1 = np.exp((reduced_planck * freq ) / (boltzmann_constant * temperature))
    n = (exp1 - 1) ** -1
    return n

def tau_displacement(atomic_mass, freq, temperature):
    #atomic_mass units ~ kg
    #freq units        ~ 1 / second

    reduced_planck = 1.05457e-34  # Joules times second
    bose_ein = bose_einstein_occupation(freq, temperature)

    m1 = 2*bose_ein + 1
    m2 = reduced_planck / (2 * atomic_mass * freq)
    disp = (m1 * m2) ** (1/2) #in meters
    disp_ang = disp * 1e10  # conversion to angstroms

    return disp

print(tau_displacement(M_c,single_phon_freq, Temp))

