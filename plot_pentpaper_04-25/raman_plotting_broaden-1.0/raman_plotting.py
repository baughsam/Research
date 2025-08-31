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

bs_Sum_cx = pd.read_csv(r'vasp_raman.dat-broaden_cx_rerun.csv')

Freq_cx = bs_Sum_cx["freq/cm-1"]

Intensity_cx = bs_Sum_cx["Intensity"]
#------------------------------------------------------------------------
##   Read in data (vdW-DF2)   ##
#------------------------------------------------------------------------
bs_Sum_df2 = pd.read_csv(r'vasp_raman.dat-broaden_df2_rerun.csv')

Freq_df2 = bs_Sum_df2["freq/cm-1"]

Intensity_df2 = bs_Sum_df2["Intensity"]
#----------------------------------------

#------------------------------------------------------------------------
##   Read in data  (vdW-cx)## original
#------------------------------------------------------------------------

bs_Sum_cx_or = pd.read_csv(r' vasp_raman.dat-broaden_cx.csv')

Freq_cx_or = bs_Sum_cx_or["freq/cm-1"]

Intensity_cx_or = bs_Sum_cx_or["Intensity"]
#------------------------------------------------------------------------
##   Read in data (vdW-DF2)   ## original
#------------------------------------------------------------------------
bs_Sum_df2_or = pd.read_csv(r'vasp_raman.dat-broaden_df2.csv')

Freq_df2_or = bs_Sum_df2_or["freq/cm-1"]

Intensity_df2_or = bs_Sum_df2_or["Intensity"]
#----------------------------------------

#------------------------------------------------------------------------
##   Read in data (PBE)   ##
#------------------------------------------------------------------------
#bs_Sum_pbe = pd.read_csv(r'vasp_raman.dat-broaden_pbe.csv')

#Freq_pbe = bs_Sum_pbe["freq/cm-1"]

#Intensity_pbe = bs_Sum_pbe["Intensity"]
#----------------------------------------
#------------------------------------------------------------------------
##  Graph data  ## 
#------------------------------------------------------------------------
System = "_Pentacene_2_reruns+or_cx"

#########################
plt.figure(2)
# mpl.rcParams["font.weight"] = "bold"
# mpl.rcParams["axes.labelweight"] = "bold"

print(Freq_cx)
plt.plot(Freq_cx, Intensity_cx, color = "#ff0096", label = r"cx", linewidth=0.5)#, linestyle = "", label = r"$a$ (cx)", markerfacecolor='none', markeredgecolor= "#ff0096") ## green
#plt.plot(Freq_df2, Intensity_df2, color = "#FF9600", label = r"DF2", linewidth=0.5)#, linestyle = "", label = r"$a$ (DF2)", markerfacecolor='none', markeredgecolor= "#FF9600") ## pink
plt.plot(Freq_cx_or, Intensity_cx_or, color = "#e900ff", label = r"cx_or", linewidth=0.5)#, linestyle = "", label = r"$a$ (cx)", markerfacecolor='none', markeredgecolor= "#ff0096") ## green
#plt.plot(Freq_df2_or, Intensity_df2_or, color = "#16ff00", label = r"DF2_or", linewidth=0.5)#, linestyle = "", label = r"$a$ (DF2)", markerfacecolor='none', markeredgecolor= "#FF9600") ## pink
#plt.plot(Freq_pbe, Intensity_pbe, color = "#004BFF", label = r"PBE", linewidth=0.5)#, linestyle = "", label = r"$a$ (PBE)", markerfacecolor='none', markeredgecolor= "#004BFF") ## blue

"""
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
"""


plt.ylabel(r"Intensity")
plt.xlabel(r"Frequency (cm$^{-1}$)")
plt.xlim(0, 250)

plt.legend(loc='center', bbox_to_anchor=(0.5, -0.30), ncol=3, handlelength = 1.2, handletextpad = 0.5, columnspacing=0.3)

plt.savefig("Raman_Spectrum"+System+".png", bbox_inches='tight')


