import numpy as np
import matplotlib.pyplot as plt

# Gaussian Distribution Function
def gaussian_dist_2d(x_pos, y_pos, spread_x, spread_y, amplitude, center_x, center_y):
    term1 = - ( (x_pos - center_x)**2 ) / (2 * spread_x**2)
    term2 = - ( (y_pos - center_y)**2 ) / (2 * spread_y**2)
    dist_funct = amplitude * np.exp( term1 + term2 )
    return dist_funct

# Initializing Gaussian Wavepack on N_x x N_y sized grid
# Real Space Dimension in nanometers
length_x = 40
length_y = 40

# Grid Dimensions
grid_x = 100
grid_y = 100

# Simulation Distances
delta_x = length_x / (grid_x-1)
delta_y = length_y / (grid_y-1)

#Box (real space) center
x_0 = length_x / 2
y_0 = length_y / 2

# Gaussian Spread (Change to actual simulation values)
sigma_x = 5
sigma_y = 5

# Amplitude
amplitude = 1

# Initialize Grid
occupation_matrix = np.empty((grid_x, grid_y))

for i in range(grid_x):
    for j in range(grid_y):
        x_i = i * delta_x
        y_i = j * delta_y
        occupation_matrix[i,j] = gaussian_dist_2d(x_pos=x_i, y_pos=y_i, spread_x=sigma_x, spread_y=sigma_y, amplitude=amplitude, center_x=x_0, center_y=y_0)




# 2. Create the heatmap
# Use cmap='Blues' for Light -> Dark
# Use cmap='Blues_r' for Dark -> Light
plt.imshow(occupation_matrix, cmap='Blues')

# 3. Add a colorbar to show the scale
plt.colorbar()

# 4. Display the plot
plt.title("Matplotlib Heatmap")
plt.show()