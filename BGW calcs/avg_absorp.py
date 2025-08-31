import numpy as np
import matplotlib.pyplot as plt


#files = ["absorption_b1_eh_PBE.dat","absorption_b2_eh_PBE.dat", "absorption_b3_eh_PBE.dat"]
#files = ["absorption_b1_eh_cx.dat","absorption_b2_eh_cx.dat", "absorption_b3_eh_cx.dat"]
files = ["absorption_b1_eh_91cm-1.dat","absorption_b2_eh_91cm-1.dat", "absorption_b3_eh_91cm-1.dat"]
colors = ["red","blue","green"] #legacy; doesn't do anything

plt.figure(figsize=(6, 6))

eps2_values = [] # A temporary list to store eps2 from each file

for fname, color in zip(files, colors):
    data = np.loadtxt(fname, comments='#')
    omega = data[:, 0]
    eps2 = data[:, 1]
    eps2_values.append(eps2) # Add eps2 from the current file to the list

# Now, calculate the average after the loop has populated eps2_values
if eps2_values: # Check if the list is not empty
    # Assuming all eps2 arrays have the same length for direct averaging
    eps2_avg = np.mean(eps2_values, axis=0)
    print("Average eps2:", eps2_avg)


plt.plot(omega, eps2_avg, color='green', label='Average eps2')


plt.xlabel('eV')
plt.ylabel('Eps2')
plt.xlim(0,10)
plt.title('91cm-1_eps2_avg')
plt.legend()
plt.savefig('pent_91cm-1_absorp_avg.png')
plt.show()
