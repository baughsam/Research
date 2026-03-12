from dataclasses import dataclass
import numpy as np

@dataclass
class UpwindDifference2d:
    # Spatial Difference
    dx: float
    dy: float

    # Velocities
    velocity_x: float = 0.0
    velocity_y: float = 0.0


    def upwindDifference(self) -> np.ndarray:
        # Raise Error: Initial array not input
        if self.initial_2d_array is None:
            raise ValueError("Error: 'initial_2d_array' is empty. Provide a 2D array before calculating.")

        # Raise Error: Initial array incorrect dimensions
        if self.initial_2d_array.ndim != 2:
            raise ValueError("Error: 'initial_2d_array' is not a 2D array.")

        # Initializing zero array for slice derivation
        print("Initializing temp arrays.")
        temp_x = np.zeros_like(self.initial_2d_array)
        temp_y = np.zeros_like(self.initial_2d_array)

        ### --- x differentiation --- ###
        if self.velocity_x > 0:
            temp_x[1:, :] = (self.initial_2d_array[1:, :] - self.initial_2d_array[:-1, :]) / self.dx
            temp_x[0, :] = (self.initial_2d_array[0, :] - 0.0) / self.dx
            self.x_deriv_array = -self.velocity_x * temp_x
        elif self.velocity_x < 0:
            temp_x[:-1, :] = (self.initial_2d_array[1:, :] - self.initial_2d_array[:-1, :]) / self.dx
            temp_x[-1, :] = (0.0 - self.initial_2d_array[-1, :]) / self.dx
            self.x_deriv_array = -self.velocity_x * temp_x
        else:
            self.x_deriv_array = temp_x
        print("x derivative complete.")

        ### --- y differentiation --- ###
        if self.velocity_y > 0:
            temp_y[:, 1:] = (self.initial_2d_array[:, 1:] - self.initial_2d_array[:, :-1]) / self.dy
            temp_y[:, 0] = (self.initial_2d_array[:, 0] - 0.0) / self.dy
            self.y_deriv_array = -self.velocity_y * temp_y
        elif self.velocity_y < 0:
            temp_y[:, :-1] = (self.initial_2d_array[:, 1:] - self.initial_2d_array[:, :-1]) / self.dy
            temp_y[:, -1] = (0.0 - self.initial_2d_array[:, -1]) / self.dy
            self.y_deriv_array = -self.velocity_y * temp_y
        else:
            self.y_deriv_array = temp_y
        print("y derivative complete.")

        self.full_derivative_array = self.x_deriv_array + self.y_deriv_array
        print("Upwind Difference Complete.")
        print("Returning differentiated array.")

        return self.full_derivative_array