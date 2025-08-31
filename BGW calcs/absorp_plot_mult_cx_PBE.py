import numpy as np
import matplotlib.pyplot as plt
#PBE = red ; -cx = vblue ; 91cm-1 = green

#files = ["absorption_b3_eh_PBE.dat", "absorption_b3_eh_cx.dat","absorption_b3_eh_91cm-1.dat"]
#files = ["absorption_b2_eh_PBE.dat", "absorption_b2_eh_cx.dat","absorption_b2_eh_91cm-1.dat"]
#files = ["absorption_b1_eh_PBE.dat", "absorption_b1_eh_cx.dat","absorption_b1_eh_91cm-1.dat"]
files = ["absorption_b3_eh_cx.dat","absorption_b3_eh_91cm-1.dat"]
colors = ["blue", "green"]
#colors = ["red", "blue", "green"]

legend_names = ["-cx", "91cm-1"]

plt.figure(figsize=(6, 6))

for fname, color, legend in zip(files, colors, legend_names):
    data = np.loadtxt(fname, comments='#')
    omega = data[:, 0]
    eps2 = data[:, 1]
    plt.plot(omega, eps2, color=color, label=legend)

plt.xlabel("eV")
plt.ylabel("eps2")
plt.ylim(0,4)
plt.xlim(1,4) #Look at energy range for the bands that I actually calculated
plt.title('absorption_b3_eh')
plt.legend()
plt.savefig('absorp_b3_cx_91cm-1_zoom4.png')
plt.show()