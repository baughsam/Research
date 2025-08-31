# minimal script for plotting multiple absorption.dat files in the same figure,
# all you would need to do would be change the file paths,
# legend names, and colors based on how many files you want to plot.

import numpy as np
import matplotlib.pyplot as plt

files = ["file_path_1", "file_path_2", "file_path_3", "file_path_4"]

colors = ["red", "green", "blue", "orange"]

legend_names = ["1v_1c", "2v_2c", "3v_3c", "4v_4c"]

plt.figure(figsize=(6, 6))

for fname, color, legend in zip(files, colors, legend_names):
    data = np.loadtxt(fname, comments='#')
    omega = data[:, 0]
    eps2 = data[:, 1]
    plt.plot(omega, eps2, color=color, label=legend)

plt.xlabel("eV")
plt.ylabel("eps2")
plt.title("")
plt.ylim(0)
plt.xlim(0,12) #Look at energy range for the bands that I actually calculated
plt.legend()
plt.show()