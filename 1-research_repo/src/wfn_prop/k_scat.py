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

@dataclass
class two_state_transition_matrix(Kscat):
    """
    Constructs the scattering matrix from first principles
    Based on the 2024 paper: Phonon-Driven Femtosecond
    "Dynamics of Excitons in Crystalline Penacene from First Principles"
    """
    # Shape: (N_Q, N_Q)
    k_BB: np.ndarray
    k_BD: np.ndarray
    gamma_decay_constant: float
    map_Q_to_q: np.ndarray
    gamma_index: int
    transition_matrix: np.ndarray = field(init=False)

    def __post_init__(self):
        """
        Builds the static K_scat matrix from the provided raw physics arrays.
        """
        self.transition_matrix = self._build_operator

    def _build_operator(self) -> np.ndarray:
        # Check matrix sizes
        N_Q, N_q = self.k_BB.shape

        assert self.k_BB.shape == self.k_BD.shape, f"FATAL: Dimension mismatch between k_BB {self.k_BB.shape} and k_BD {self.k_BD.shape}"
        assert self.map_Q_to_q == (N_Q, N_q), "FATAL: Mapping array shape does not match the rate arrays."

        print(f"Inititalizing two_state_transition_matrix operator of size ({N_Q}, {N_Q})...")

        # Initialize empty matrix
        K_scat = np.zeros((N_Q, N_Q))

        # Build diagonal (Losses)
        sum_k_BB = np.sum(self.k_BB, axis=1)
        sum_k_BD = np.sum(self.k_BD, axis=1)

        # Build off-diagonals (Gains)
        for i in range(N_Q):
            for j in range(N_q):
                rate = self.k_BB[i,j]
                Q_final = self.map_Q_to_q[i, j]
                K_scat[Q_final, i] += rate

        # Put diagonal in K_scat
        for i in range(N_Q):
            K_scat[i,i] = - (sum_k_BB[i] + sum_k_BD[i])

        # Insert radiative decay at gamma
        K_scat[self.gamma_index, self.gamma_index] -= self.gamma_decay_constant

        return K_scat

    def calc_scattering(self, array_3d: np.ndarray) -> np.ndarray:
        """
        Executes the scattering step during the RK4 loop using Einstein summation.
        """
        if array_3d is None:
            raise ValueError("Error: 'array_3d' is empty.")

        scattered_array = np.einsum('ij, xyi -> xyj', self.transition_matrix, array_3d)
        return scattered_array