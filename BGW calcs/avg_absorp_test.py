import numpy as np
import matplotlib.pyplot as plt

files = ["test_file_1.txt","test_file_2.txt","test_file_3.txt"]
#files = ["absorption_b1_eh_PBE.dat","absorption_b2_eh_PBE.dat", "absorption_b3_eh_PBE.dat"]
colors = ["red","blue","green"] # You might not need all these colors for just two files
#legend_names = ["1v_1c", "2v_2c"] # Adjust as needed for your specific plotting

plt.figure(figsize=(6, 6))

eps2_values = [] # A temporary list to store eps2 from each file
print(eps2_values)
for fname, color in zip(files, colors):
    data = np.loadtxt(fname, comments='#')
    # print(data) # Uncomment for debugging if needed
    omega = data[:, 0]
    eps2 = data[:, 1]
    eps2_values.append(eps2) # Add eps2 from the current file to the list

print(eps2_values)
# Now, calculate the average after the loop has populated eps2_values
if eps2_values: # Check if the list is not empty
    # Assuming all eps2 arrays have the same length for direct averaging
    eps2_avg = np.mean(eps2_values, axis=0)
    print("Average eps2:", eps2_avg)


plt.plot(omega, eps2_avg, color='black', linestyle='--', label='Average eps2')


plt.xlabel('Omega')
plt.ylabel('Eps2')
plt.title('Average eps2_avg')
plt.legend()
plt.grid(True)
plt.show()