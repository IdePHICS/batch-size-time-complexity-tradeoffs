import numpy as np
from numpy.linalg import inv as inverse_matrix
from tqdm import tqdm

from .base import GiantStepBase
from .cython_erf_erf import erf_error
from .staircase_overlaps import Hermite2Relu_Staricase2
from .poly_poly import H2H2Overlaps, H3H3Overlaps, H4H4Overlaps

class GradientDescent(GiantStepBase):
    def __init__(self,
                 target: callable, W_target: np.array, n: int,
                 activation: callable, W0: np.array, a0: np.array,activation_derivative: callable,
                 gamma: float, noise: float,
                 predictor_interaction: bool = True,
                 second_layer_update: bool = False,
                 resample_every: int = 1,
                 seed: int = 0, test_size = None,
                 analytical_error = None, lazy_memory = False):
        super().__init__(target, W0.shape[0], W_target.shape[0], activation, a0, activation_derivative, gamma, noise, predictor_interaction, second_layer_update, lazy_memory)

        self.d = W0.shape[1]
        self.n = n
        self.rng = np.random.default_rng(seed) 
        self.W_target = W_target.copy()
        if not lazy_memory:
            self.W_s = [W0.copy()]
        else: 
            self._lastW = W0.copy()

        self.Ms = []
        self.Qs = []
        self.P = W_target @ W_target.T
        self.resample_every = resample_every

        self.zs, self.ys = self.samples(self.n)

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
        elif self.analytical_error == 'H4H4':
            return H4H4Overlaps(self.W_target @ self.W_target.T, self.W @ self.W_target.T, self.W @ self.W.T, self.a, self.gamma, self.noise).error()
        elif self.analytical_error == 'hermite2ReLuStaircase2':
            return Hermite2Relu_Staricase2(self.W_target @ self.W_target.T, self.W @ self.W_target.T, self.W @ self.W.T, self.a, self.gamma, self.noise).error()
        elif self.analytical_error == 'skip':
            return 0
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
        if self.lazy_memory:
            return self._lastW
        return self.W_s[-1]

    
    def _weight_loss_gradient(self, zs, ys):
        if self.predictor_interaction:
            displacements = ys - np.apply_along_axis(self.network, -1, zs @ self.W.T)
        else:
            displacements = ys
        return 1/(self.n*self.p) * np.einsum('j,uj,u,ui->ji',self.a,self.activation_derivative(zs @ self.W.T),displacements,zs)

    def _update_weight(self, zs, ys):
        return  self.gamma * self._weight_loss_gradient(zs, ys)
    
    def _new_weight(self,zs,ys):
        return self.W + self._update_weight(self.zs, self.ys)
    
    def update(self, zs, ys):
        if self.lazy_memory:
            self._lastW = self._new_weight(self.zs, self.ys)
        else:
            self.W_s.append(
                self._new_weight(self.zs, self.ys)
            )
        if self.second_layer_update:
            raise NotImplementedError
        else:
            if self.lazy_memory:
                pass
            else:
                self.a_s.append(self.a)

        GiantStepBase.update(self)

    def measure(self, zs = None, ys = None):
        self.test_errors.append(self.error(zs, ys))
        self.Ms.append(self.W @ self.W_target.T)
        self.Qs.append(self.W @ self.W.T)

    def train(self, steps, verbose=False):
        for step in tqdm(range(steps), disable=not verbose, mininterval=2):
            if self.resample_every > 0 and step % self.resample_every == 0:
                self.zs, self.ys = self.samples(self.n)
            self.update(self.zs, self.ys)
            self.measure()

class ProjectedGradientDescent(GradientDescent):
    def _new_weight(self, zs, ys):
        updateW = GradientDescent._update_weight(self, zs, ys)
        return (self.W + updateW) / np.linalg.norm(self.W + updateW, axis=1, keepdims=True)
    def _update_weight(self, zs, ys):
        return self._new_weight(zs,ys) - self.W

class SphericalGradientDescent(GradientDescent):
    def _new_weight(self, zs, ys):
        updateW = GradientDescent._update_weight(self, zs, ys)
        current_weight_norm = np.linalg.norm(self.W, axis=1) # shape (p,)
        spherical_updateW = np.einsum(
            'ja,jab->jb',
            updateW,
            (np.repeat(np.eye(self.d)[np.newaxis,:, :], self.p, axis=0) - np.einsum('ja,jb,j->jab', self.W, self.W, 1/current_weight_norm**2))
        )
        return (self.W + spherical_updateW) / np.linalg.norm(self.W + spherical_updateW, axis=1, keepdims=True)
    def _update_weight(self, zs, ys):
        return self._new_weight(zs,ys) - self.W
    
class SAM(GradientDescent):
    def __init__(self,
                 target: callable, W_target: np.array, n: int,
                 activation: callable, W0: np.array, a0: np.array, activation_derivative: callable,
                 gamma: float, noise: float, rho_prefactor: float,
                 predictor_interaction: bool = True,
                 second_layer_update: bool = False,
                 resample_every: int = 1,
                 seed: int = 0, test_size = None,
                 analytical_error = None, lazy_memory = False):
        super().__init__(target, W_target, n, activation, W0, a0, activation_derivative, gamma, noise, predictor_interaction, second_layer_update, resample_every, seed, test_size, analytical_error, lazy_memory)
        self.rho = rho_prefactor/ self.d 
    def _weight_loss_gradient(self, zs, ys):
        def minusgradW(Wtilde):
            if self.predictor_interaction:
                displacements = ys - np.apply_along_axis(self.network, -1, zs @ Wtilde.T)
            else:
                displacements = ys
            return 1/(self.n*self.p) * np.einsum('j,uj,u,ui->ji',self.a,self.activation_derivative(zs @ Wtilde.T),displacements,zs)
        return  minusgradW(self.W + self.rho * minusgradW(self.W)) ### TO fix: a) rho; b) lambda function gradW useless; c) make sure grad is minus gradient
    
class ProjectedSAM(SAM, ProjectedGradientDescent):
    # _weight_loss_gradient = SAM._weight_loss_gradient
    pass

class SphericalSAM(SAM, SphericalGradientDescent):
    # _weight_loss_gradient = SAM._weight_loss_gradient
    pass