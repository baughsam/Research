#name: Tina Mihm
#Date: Jan 16, 2025
#Description: Code reads in output csv from modes_to_vesta.py and graphs the displacements along different lattice vectors. 


import numpy as np
import subprocess
import re
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pylab import *
from matplotlib.colors import *
from matplotlib.font_manager import fontManager, FontProperties
from scipy.optimize import curve_fit
from matplotlib.ticker import MaxNLocator

#from ase.io import read, write
#from ase.io.jsonio import read_json
import matplotlib.pyplot as plt

#-------------------------------------------------------------------------------------------
###
### SET UP FIGURE
###
# rcParams['text.usetex'] = True
###
### Fonts
###
rcParams['axes.labelsize'] = 6
rcParams['xtick.labelsize'] = 6
rcParams['ytick.labelsize'] = 6
rcParams['legend.fontsize'] = 8
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = 'Computer Modern Roman'
rcParams['legend.numpoints'] = 1
rcParams['lines.linewidth'] = 1.0
rcParams['lines.markersize'] = 4.0
# http://stackoverflow.com/questions/7906365/matplotlib-savefig-plots-different-from-show
rcParams['savefig.dpi'] = 400
rcParams['figure.dpi'] = 600
###
### Size of the figure
###
ratio=(np.sqrt(5)-1)/2.0    # golden ratio
# #ratio=1                     # square figure
plt.rcParams["figure.figsize"] = 3.37, 3.37*ratio
# rcParams[‘figure.figsize’] = 3.37, 3.37
fig = figure()

#------------------------------------------------------------------------
##   Read in data  (vdW-cx)##
#------------------------------------------------------------------------

bs_Sum_cx = pd.read_csv(r'1_Summed_over_all-atoms_Eignfrequency_info_cx.csv')

Freq_cx = bs_Sum_cx["Eigenfrequency (cm-1)"]

X_cx = bs_Sum_cx["Eigenvector (dx)"]

Y_cx = bs_Sum_cx["Eigenvector (dy)"]

Aproj_cx = bs_Sum_cx["A projection"]

Bproj_cx = bs_Sum_cx["B projection"]

abs_Aproj_cx = bs_Sum_cx["abs A projection"]

abs_Bproj_cx = bs_Sum_cx["abs B projection"]

#------------------------------------------------------------------------
##   Read in data (vdW-DF2)   ##
#------------------------------------------------------------------------
bs_Sum_df2 = pd.read_csv(r'2_Summed_over_all-atoms_Eignfrequency_info_df2.csv')

Freq_df2 = bs_Sum_df2["Eigenfrequency (cm-1)"]

X_df2 = bs_Sum_df2["Eigenvector (dx)"]

Y_df2 = bs_Sum_df2["Eigenvector (dy)"]

Aproj_df2 = bs_Sum_df2["A projection"]

Bproj_df2 = bs_Sum_df2["B projection"]

abs_Aproj_df2 = bs_Sum_df2["abs A projection"]

abs_Bproj_df2 = bs_Sum_df2["abs B projection"]

#------------------------------------------------------------------------
##   Read in data (PBE)   ##
#------------------------------------------------------------------------
bs_Sum_pbe = pd.read_csv(r'3_Summed_over_all-atoms_Eignfrequency_info_pbe.csv')

Freq_pbe = bs_Sum_pbe["Eigenfrequency (cm-1)"]

X_pbe = bs_Sum_pbe["Eigenvector (dx)"]

Y_pbe = bs_Sum_pbe["Eigenvector (dy)"]

Aproj_pbe = bs_Sum_pbe["A projection"]

Bproj_pbe = bs_Sum_pbe["B projection"]

abs_Aproj_pbe = bs_Sum_pbe["abs A projection"]

abs_Bproj_pbe = bs_Sum_pbe["abs B projection"]
#------------------------------------------------------------------------
##  Graph data  ## 
#------------------------------------------------------------------------
System = "_Pentacene_3"
First_real = 3
Mode_200 = 22

#cx
Freq_cx = Freq_cx[First_real:Mode_200]
abs_Aproj_cx = abs_Aproj_cx[First_real:Mode_200]
abs_Bproj_cx = abs_Bproj_cx[First_real:Mode_200]

#vdw-df2
Freq_df2 = Freq_df2[First_real:Mode_200]
abs_Aproj_df2 = abs_Aproj_df2[First_real:Mode_200]
abs_Bproj_df2 = abs_Bproj_df2[First_real:Mode_200]

#pbe
Freq_pbe = Freq_pbe[First_real:Mode_200]
abs_Aproj_pbe = abs_Aproj_pbe[First_real:Mode_200]
abs_Bproj_pbe = abs_Bproj_pbe[First_real:Mode_200]

#X = X[First_real:Mode_200]
#Y = Y[First_real:Mode_200]
#Aproj = Aproj[First_real:Mode_200]
#Bproj = Bproj[First_real:Mode_200]


#ab_ratio = Aproj/Bproj
#abs_ab_ratio = abs_Aproj/abs_Bproj

#########################
plt.figure(2)
# mpl.rcParams["font.weight"] = "bold"
# mpl.rcParams["axes.labelweight"] = "bold"


plt.plot(Freq_cx, abs_Aproj_cx, linestyle = "", marker = "^", label = r"$a$ (cx)", markerfacecolor='none', markeredgecolor= "#ff0096") ## green
plt.plot(Freq_df2, abs_Aproj_df2, linestyle = "", marker = "^", label = r"$a$ (DF2)", markerfacecolor='none', markeredgecolor= "#FF9600") ## pink
plt.plot(Freq_pbe, abs_Aproj_pbe, linestyle = "", marker = "^", label = r"$a$ (PBE)", markerfacecolor='none', markeredgecolor= "#004BFF") ## blue

plt.plot(Freq_cx, abs_Bproj_cx, linestyle = "", marker = "o", label = r"$b$ (cx)", markerfacecolor='none', markeredgecolor= "#ff0096") ## purple
plt.plot(Freq_df2, abs_Bproj_df2, linestyle = "", marker = "o", label = r"$b$ (DF2)", markerfacecolor='none', markeredgecolor= "#FF9600") ## orange
plt.plot(Freq_pbe, abs_Bproj_pbe, linestyle = "", marker = "o", label = r"$b$ (PBE)", markerfacecolor='none', markeredgecolor= "#004BFF") ## blue


###Dotted lines
plt.axvline(x=37, color='black', linestyle='--')
plt.annotate('37', (37, 1.0), xytext=(38, 1.1), fontsize=5)

plt.axvline(x=63, color='black', linestyle='--')
plt.annotate('63', (63, 1.0), xytext=(64, 1.1), fontsize=5)

plt.axvline(x=91, color='black', linestyle='--')
plt.annotate('91', (91, 1.0), xytext=(92, 1.1), fontsize=5)

plt.axvline(x=161, color='black', linestyle='--')
plt.annotate('161', (161, 1.0), xytext=(162, 1.1), fontsize=5)

plt.axvline(x=176, color='black', linestyle='--')
plt.annotate('176', (176, 1.0), xytext=(177, 1.1), fontsize=5)
###



plt.ylabel(r"Abs Displacement (fractional coord)")
plt.xlabel(r"Frequency mode (cm$^{-1}$)")
plt.ylim(0, 1.2)

plt.legend(loc='center', bbox_to_anchor=(0.5, -0.40), ncol=2, handlelength = 0.8, handletextpad = 0.5, columnspacing=0.3)

plt.savefig("1_Absolute_displacement_along_lattice"+System+".png", bbox_inches='tight')


