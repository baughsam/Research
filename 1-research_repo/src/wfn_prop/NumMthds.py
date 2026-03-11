from dataclasses import dataclass
import numpy as np

@dataclass
class UpwindDifference2d:



    def upwindDifference(self) -> np.ndarray:
        # Raise Error: Initial array not input
        if self.initial_2d_array is None:
            raise ValueError("Error: 'initial_2d_array' is empty. Provide a 2D array before calculating.")

        # Raise Error: Initial array incorrect dimensions
        if self.initial_2d_array.ndim != 2:
            raise ValueError("Error: 'initial_2d_array' is not a 2D array.")

        # Initializing zero array for slice derivation
        temp_x = np.zeros_like(self.initial_2d_array)
        temp_y = np.zeros_like(self.initial_2d_array)

        # x differentiation
        if self.velocity_x > 0:
            temp_x[1:, :] = (self.initial_2d_array[1:, :] - self.initial_2d_array[:-1,]) / self.dx
            temp_x[:1, :] = (self.initial_2d_array[1:, :] - 0.0) / self.dx
            self.x_deriv_array = -self.velocity_x * temp_x
        elif self.velocity_x < 0:
            temp_x[:-1, :] = (self.initial_2d_array[1:, :] - self.initial_2d_array[:-1, :]) / self.dx
            temp_x[-1:, :] = (self.initial_2d_array[1:, :] - 0.0) / self.dx
            self.x_deriv_array = -self.velocity_x * temp_x
        else:
            temp_x = self.initial_2d_array