import numpy as np
from numpy.linalg import inv as inverse_matrix  
from tqdm import tqdm

class GiantStepBase():
    _adaptive_percentage_threshold = 0.6
    _adaptive_lr_decay = .995
    _adaptive_switch_lr_jump = .1
    def __init__(self,
                 target: callable, p: int, k: int,
                 activation: callable, a0: np.array, activation_derivative: callable,
                 gamma: float, noise: float,
                 predictor_interaction: bool, second_layer_update: bool, lazy_memory = False
                ):
        self.target = target
        self.activation = activation
        self.activation_derivative = activation_derivative
        self.gamma = gamma
        self.p = p
        self.k = k
        self.noise = noise
        self.second_layer_update = second_layer_update
        self.lazy_memory = lazy_memory
        if isinstance(predictor_interaction, bool):
            self.predictor_interaction = predictor_interaction
            self.adaptive_predictor_interaction = False
        elif predictor_interaction.lower() == 'adaptive':
            self.predictor_interaction = False
            self.adaptive_predictor_interaction = True
        else:
            raise ValueError('predictor_interaction must be bool or "adaptive"')
        assert(a0.shape == (p,))
        if not lazy_memory:
            self.a_s = [a0.copy()]
        else:
            self._last_a = a0.copy()
        self.test_errors = []

        self.network = lambda local_field: 1/self.p * np.dot(self.a, self.activation(local_field))

    @property
    def a(self):
        if self.lazy_memory:
            return self._last_a
        return self.a_s[-1]
    
    def update(self):
        if self.adaptive_predictor_interaction and self.test_errors[-1] < self._adaptive_percentage_threshold * self.test_errors[0]:
            if not self.predictor_interaction:
                self.gamma *= self._adaptive_switch_lr_jump
            self.predictor_interaction = True
            self.gamma *= self._adaptive_lr_decay
            # self.adaptive_predictor_interaction = False
    
class OverlapsBase(GiantStepBase):
    def __init__(self,
                 target: callable, activation: callable, activation_derivative: callable,
                 P: np.array, M0: np.array, Q0: np.array,  a0: np.array, 
                 gamma: float, noise: float,
                 I4_diagonal: bool, I4_offdiagonal: bool, I3: bool, 
                 predictor_interaction: bool, second_layer_update: bool
                ):
        super().__init__(target,Q0.shape[0],P.shape[0], activation, a0, activation_derivative, gamma, noise, predictor_interaction, second_layer_update)

        self.I3 = I3

        if isinstance(I4_diagonal, bool):
            self.I4_diagonal = I4_diagonal
            if self.I4_diagonal:
                self._invalpha = 1.
            else:
                self._invalpha = 0.
        elif isinstance(I4_diagonal, float):
            self.I4_diagonal = True
            self._invalpha = I4_diagonal
        else:
            raise TypeError('I4_diagonal must be bool or float')
        
        self.I4_offdiagonal = I4_offdiagonal

        self.Ms = [M0.copy()]
        self.Qs = [Q0.copy()]
        self.P = P.copy()
        self.inverse_P = inverse_matrix(P)

        self.measure()

    @property
    def M(self):
        return self.Ms[-1]
    
    @property
    def Q(self):
        return self.Qs[-1]
    
    def train(self, steps, verbose=False):
        for _ in tqdm(range(steps), disable=not verbose):
            self.update()
            self.measure()
            super().update()

    def overlap_update(self):
        expected_value_target, expected_value_network, expected_I4 = self.compute_expected_values()

        updateM = np.zeros(shape=(self.p, self.k))
        updateQ = np.zeros(shape=(self.p, self.p))

        if self.I3:
            updateM += self.gamma/self.p * np.einsum('j,jr->jr', self.a, expected_value_target)
            updateQ += (self.gamma/self.p) * (
                    np.einsum('j,jl->jl', self.a, expected_value_network) +
                    np.einsum('l,lj->jl', self.a, expected_value_network)
                )
        if self.I4_diagonal:
            updateQ += (self.gamma/self.p)**2 * self._invalpha * np.einsum('j,l->jl', self.a, self.a) * expected_I4

        if self.I4_offdiagonal:
            inverse_Qbot = inverse_matrix(self.Q - self.M @ self.inverse_P @ self.M.T)
            expected_value_orthogonal = expected_value_network - self.M @ self.inverse_P @ (expected_value_target.T)
            updateQ += (self.gamma/self.p)**2 * np.einsum('j,l->jl', self.a, self.a) * (
                np.einsum('jr,rt,lt->jl', expected_value_target, self.inverse_P, expected_value_target) +
                np.einsum('jm,mn,ln->jl', expected_value_orthogonal, inverse_Qbot, expected_value_orthogonal)
            )
        
        return updateM, updateQ

    def update(self):
        updateM, updateQ = self.overlap_update()
        self.Ms.append(self.M + updateM)
        self.Qs.append(self.Q + updateQ)
        if self.second_layer_update:
            raise NotImplementedError
        else:
            self.a_s.append(self.a)

    # abstact method
    def error(self):
        raise NotImplementedError('error() must be reimplemented!')
    
    def measure(self):
        self.test_errors.append(self.error())

    
class SpecializedOverlapsBase(OverlapsBase):
    def __init__(self, P: np.array, M0: np.array, Q0: np.array, a0: np.array, gamma: float, noise: float, I4_diagonal: bool=True, I4_offdiagonal: bool=True, I3: bool=True, predictor_interaction: bool=True, second_layer_update: bool=False):
        super().__init__(self._target, self._activation, self._activation_derivative, P, M0, Q0, a0, gamma, noise, I4_diagonal, I4_offdiagonal, I3, predictor_interaction, second_layer_update)


class ProjectedOverlapsBase(OverlapsBase):
    def update(self):
        updateM, updateQ = self.overlap_update()
        M = self.M
        Q = self.Q
        diagonal_updateQ = np.diagonal(updateQ)
        M_norm = np.tile(diagonal_updateQ, (self.k, 1)).T
        Q_norm = np.tile(diagonal_updateQ, (self.p, 1))
        self.Ms.append((M + updateM) / np.sqrt(1+M_norm))
        self.Qs.append((Q + updateQ) / (np.sqrt(1+Q_norm)*np.sqrt(1+Q_norm.T)))

        if self.second_layer_update:
            raise NotImplementedError
        else:
            self.a_s.append(self.a)