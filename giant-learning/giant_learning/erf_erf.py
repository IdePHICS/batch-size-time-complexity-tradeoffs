import numpy as np
from scipy.special import erf

from .base import SpecializedOverlapsBase, ProjectedOverlapsBase
from .cython_erf_erf import erf_updates, erf_error


class ErfErfOverlaps(SpecializedOverlapsBase):
    @staticmethod
    def _target(local_fields):
        return np.mean(erf(local_fields/np.sqrt(2)),axis=-1)
    @staticmethod
    def _activation(x):
        return erf(x/np.sqrt(2))
    @staticmethod
    def _activation_derivative(x):
        return np.sqrt(2/np.pi) * np.exp(-x**2/2)
    
    def compute_expected_values(self):
        return erf_updates(self.Q, self.M, self.P, self.a, self.noise)
    
    def error(self):
        return erf_error(self.Q, self.M, self.P, self.a, self.noise)
    
class ProjectedErfErfOverlaps(ProjectedOverlapsBase, ErfErfOverlaps):
    pass
        
