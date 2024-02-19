import numpy as np
from numpy.linalg import inv as inverse_matrix
from tqdm import tqdm

from .base import GiantStepBase
from .cython_erf_erf import erf_error
from .staircase_overlaps import Hermite2Relu_Staricase2
from .poly_poly import H2H2Overlaps, H3H3Overlaps

class GradientDescent(GiantStepBase):
    def __init__(self,
                 target: callable, W_target: np.array, n: int,
                 activation: callable, W0: np.array, a0: np.array,activation_derivative: callable,
                 gamma: float, noise: float,
                 predictor_interaction: bool=True,
                 second_layer_update: bool=False,
                 resampling: bool = True,
                 seed: int = 0, test_size = None,
                 analytical_error = None):
        super().__init__(target, W0.shape[0], W_target.shape[0], activation, a0, activation_derivative, gamma, noise, predictor_interaction, second_layer_update)

        self.d = W0.shape[1]
        self.n = n
        self.rng = np.random.default_rng(seed)

        self.W_target = W_target.copy()
        self.W_s = [W0.copy()]

        self.resampling = resampling
        if not resampling:
            self.fixed_zs, self.fixed_ys = self.samples(self.n)

        self.analytical_error = analytical_error
        if self.analytical_error is None:
            raise Warning('No analytical error specified, using numerical error. Can be slow!')
            if test_size is None:
                self.test_size = self.n
            self.test_size = test_size
            self.zs_test, self.ys_test = self.samples(self.test_size)
        self.measure()

    def samples(self, size):
        zs = self.rng.normal(size=(size,self.d))
        ys = np.apply_along_axis(self.target, -1, zs @ self.W_target.T) + np.sqrt(self.noise) * np.random.normal(size=(size,))
        return zs, ys

    def error(self, zs, ys):
        if self.analytical_error == 'erferf':
            return erf_error(self.W @ self.W.T, self.W @ self.W_target.T, self.W_target @ self.W_target.T, self.a, self.noise)
        elif self.analytical_error == 'H2H2':
            return H2H2Overlaps(self.W_target @ self.W_target.T, self.W @ self.W_target.T, self.W @ self.W.T, self.a, self.gamma, self.noise).error()
        elif self.analytical_error == 'H3H3':
            return H3H3Overlaps(self.W_target @ self.W_target.T, self.W @ self.W_target.T, self.W @ self.W.T, self.a, self.gamma, self.noise).error()
        elif self.analytical_error == 'hermite2ReLuStaircase2':
            return Hermite2Relu_Staricase2(self.W_target @ self.W_target.T, self.W @ self.W_target.T, self.W @ self.W.T, self.a, self.gamma, self.noise).error()
        elif self.analytical_error is None:
            if zs is None and ys is None:
                zs, ys = self.zs_test, self.ys_test
            return 1/2 * np.mean(
                (np.apply_along_axis(self.network, -1, zs @ self.W.T)- ys)**2
            )
        else:
            raise ValueError('Unknown value analytical error')
    
    @property
    def W(self):
        return self.W_s[-1]

    def weight_update(self, zs, ys):
        if self.predictor_interaction:
            displacements = ys - np.apply_along_axis(self.network, -1, zs @ self.W.T)
        else:
            displacements = ys

        return self.gamma * 1/(self.n*self.p) * np.einsum('j,uj,u,ui->ji',self.a,self.activation_derivative(zs @ self.W.T),displacements,zs)

    def update(self, zs, ys):
        updateW = self.weight_update(zs, ys)
        self.W_s.append(
            self.W + updateW
        )
        if self.second_layer_update:
            raise NotImplementedError
        else:
            self.a_s.append(self.a)

        GiantStepBase.update(self)

    def measure(self, zs = None, ys = None):
        self.test_errors.append(self.error(zs, ys))

    def train(self, steps, verbose=False):
        for step in tqdm(range(steps), disable=not verbose):
            if self.resampling:
                zs, ys = self.samples(self.n)
                self.update(zs, ys)
            else:
                self.update(self.fixed_zs, self.fixed_ys)
            self.measure()

class SphericalGradientDescent(GradientDescent):
    def update(self, zs, ys):
        updateW = self.weight_update(zs, ys)
        self.W_s.append(
            (self.W + updateW) / np.linalg.norm(self.W + updateW, axis=1, keepdims=True)
        )
        if self.second_layer_update:
            raise NotImplementedError
        else:
            self.a_s.append(self.a)

        GiantStepBase.update(self)