import numpy as np
import matplotlib.pyplot as plt


#files = ["absorption_b1_eh_PBE.dat","absorption_b2_eh_PBE.dat", "absorption_b3_eh_PBE.dat"]
#files = ["absorption_b1_eh_cx.dat","absorption_b2_eh_cx.dat", "absorption_b3_eh_cx.dat"]
files_1 = ["absorption_b1_eh_91cm-1.dat","absorption_b2_eh_91cm-1.dat", "absorption_b3_eh_91cm-1.dat"]
files_2 = ["absorption_b1_eh_cx.dat","absorption_b2_eh_cx.dat", "absorption_b3_eh_cx.dat"]
colors = ["red","blue","green"] #legacy; doesn't do anything

plt.figure(figsize=(6, 6))

eps2_values_1 = [] # A temporary list to store eps2 from each file
eps2_values_2 = [] # A temporary list to store eps2 from each file

for fname1, color in zip(files_1, colors):
    data = np.loadtxt(fname1, comments='#')
    omega1 = data[:, 0]
    eps2_1 = data[:, 1]
    eps2_values_1.append(eps2_1) # Add eps2 from the current file to the list

for fname2, color in zip(files_2, colors):
    data = np.loadtxt(fname2, comments='#')
    omega2 = data[:, 0]
    eps2_2 = data[:, 1]
    eps2_values_2.append(eps2_2) # Add eps2 from the current file to the list

# Now, calculate the average after the loop has populated eps2_values
if eps2_values_1 and eps2_values_2: # Check if the list is not empty
    # Assuming all eps2 arrays have the same length for direct averaging
    eps2_avg_1 = np.mean(eps2_values_1, axis=0)
    eps2_avg_2 = np.mean(eps2_values_2, axis=0)
    #print("Average eps2:", eps2_avg)


plt.plot(omega1, eps2_avg_1, color='green', label='91cm-1')
plt.plot(omega2, eps2_avg_2, color='blue', label='pristine')


plt.xlabel('$\omega$ (eV)')
plt.ylabel('$\epsilon_2$ ($\omega$)')
plt.xlim(1,4)
plt.ylim(0,1.75)
plt.yticks([])
plt.title('Avg $\epsilon_2$')
plt.legend()
plt.savefig('pent_-cx_91cm-1_absorp_avg.png')
plt.show()
