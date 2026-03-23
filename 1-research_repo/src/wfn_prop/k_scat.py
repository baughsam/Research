from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import numpy as np

class Kscat(ABC):
    @abstractmethod
    def calc_scattering(self):
        pass

@dataclass()
class decay(Kscat):
    scat_time: float = 0.5 #femtoseconds???

    def calc_scattering(self, array_2d: np.ndarray) -> np.ndarray:
        if array_2d is None:
            raise ValueError("Error: 'array_2d' is empty. Provide a 2D array before calculating.")

        time_const = 1 / self.scat_time

        #Communication is key!
        print("Calculating decay...")
        decay_scat_array = - time_const * array_2d

        return decay_scat_array



