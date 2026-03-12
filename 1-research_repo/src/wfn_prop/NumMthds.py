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


    def upwindDifference(self, array_2d: np.ndarray) -> np.ndarray:
        # Raise Error: Initial array not input
        if array_2d is None:
            raise ValueError("Error: 'initial_2d_array' is empty. Provide a 2D array before calculating.")

        # Raise Error: Initial array incorrect dimensions
        if array_2d.ndim != 2:
            raise ValueError(f"Error: Expected a 2D array, but got a {array_2d.ndim}D array.")

        # Initializing zero array for slice derivation
        print("Initializing temp arrays.")
        temp_x = np.zeros_like(array_2d)
        temp_y = np.zeros_like(array_2d)

        ### --- x differentiation --- ###
        if self.velocity_x > 0:
            temp_x[1:, :] = (array_2d[1:, :] - array_2d[:-1, :]) / self.dx
            temp_x[0, :] = (array_2d[0, :] - 0.0) / self.dx
            x_deriv_array = -self.velocity_x * temp_x
        elif self.velocity_x < 0:
            temp_x[:-1, :] = (array_2d[1:, :] - array_2d[:-1, :]) / self.dx
            temp_x[-1, :] = (0.0 - array_2d[-1, :]) / self.dx
            x_deriv_array = -self.velocity_x * temp_x
        else:
            x_deriv_array = temp_x
        print("x derivative complete.")

        ### --- y differentiation --- ###
        if self.velocity_y > 0:
            temp_y[:, 1:] = (array_2d[:, 1:] - array_2d[:, :-1]) / self.dy
            temp_y[:, 0] = (array_2d[:, 0] - 0.0) / self.dy
            y_deriv_array = -self.velocity_y * temp_y
        elif self.velocity_y < 0:
            temp_y[:, :-1] = (array_2d[:, 1:] - array_2d[:, :-1]) / self.dy
            temp_y[:, -1] = (0.0 - array_2d[:, -1]) / self.dy
            y_deriv_array = -self.velocity_y * temp_y
        else:
            y_deriv_array = temp_y
        print("y derivative complete.")

        full_derivative_array = x_deriv_array + y_deriv_array
        print("Upwind Difference Complete.")
        print("Returning differentiated array.")

        return full_derivative_array

@dataclass
class RungeKutta4_2d:
    # Initial Array
    initial_2d_array: np.ndarray | None

    # Timestep
    dt: float

    def RK4(self, ) -> np.ndarray:
