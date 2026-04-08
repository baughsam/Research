from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import numpy as np

class Kscat(ABC):
    @abstractmethod
    def calc_scattering(self):
        pass

@dataclass()
class Decay(Kscat):
    scat_time: float = 0.5 #femtoseconds???

    def calc_scattering(self, array_2d: np.ndarray) -> np.ndarray:
        if array_2d is None:
            raise ValueError("Error: 'array_2d' is empty. Provide a 2D array before calculating.")

        time_const = 1 / self.scat_time

        #Communication is key!
        print("Calculating decay...")
        decay_scat_array = -time_const * array_2d

        return decay_scat_array

@dataclass()
class FickDiff(Kscat):
    D_coeff: float # Diffustion coefficient #nm^2 /fs (I think)
    dx: float      # Grid spacing in x
    dy: float      # Grid spacing in y

    def calc_scattering(self, array_2d: np.ndarray) -> np.ndarray:
        if array_2d is None:
            raise ValueError("Error: 'array_2d' is empty. Provide a 2D array before calculating.")

        # Empty array
        diff_scat_array = np.zeros_like(array_2d)

        # 2nd derivative in x
        d2n_dx2 =(array_2d[2:,:] - 2*array_2d[1:-1, :] + array_2d[:-2,:]) / (self.dx**2)

        #2nd derivative in y
        d2n_dy2 = (array_2d[:, 2:] - 2 * array_2d[:, 1:-1] + array_2d[:, :-2]) / (self.dy ** 2)

        diff_scat_array[1:-1, :] += d2n_dx2
        diff_scat_array[:, 1:-1] += d2n_dy2

        return self.D_coeff * diff_scat_array

@dataclass
class PhononScat(Kscat):
    # Shape: (Nq, Nq)
    # Precalculated 2D Matrix where element [i,j] is the rate of scattering from state i to state j
    transition_matrix: np.ndarray

    def calc_scattering(self, array_3d: np.ndarray) -> np.ndarray:
        # array_3d has shape: (Nx, Ny, Nq)
        # We are multiplying the transition matrix against the last axis (Nq)
        # Using Einstein Notation and a method called .einsum
        # 'ij' is the transition matrix (from i to j)
        # 'xyi' is the 3D tensor (x, y, initial state i)
        # 'xyj' is the output 3D tensor (x, y, final state j)

        scattered_array = np.einsum('ij, xyi -> xyj', self.transition_matrix, array_3d)

        return scattered_array