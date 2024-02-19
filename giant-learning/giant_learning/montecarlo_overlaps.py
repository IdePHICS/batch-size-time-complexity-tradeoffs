import numpy as np
from tqdm import tqdm
from collections.abc import Iterable

from .base import OverlapsBase

class MonteCarloOverlaps(OverlapsBase):
    def __init__(self,
                 target: callable, activation: callable, activation_derivative: callable,
                 P: np.array, M0: np.array, Q0: np.array,  a0: np.array, 
                 gamma: float, noise: float,
                 I4_diagonal: bool = True, I4_off_diagonal: bool = True, second_layer_update: bool = False, 
                 seed: int = 0, mc_size = None):
        self.rng = np.random.default_rng(seed)
        if mc_size is None:
            self.mc_size = self.n
        self.mc_size = mc_size

        super().__init__(target, activation, activation_derivative, P, M0, Q0, a0, gamma, noise, I4_diagonal, I4_off_diagonal, second_layer_update)
    
    def local_fields_montecarlo(self, fs, std = False):
        single_input = False
        if not isinstance(fs, Iterable):
            fs = [fs]
            single_input = True

        Omega = np.block([[self.Q, self.M], [self.M.T, self.P]])
        local_fields_samples = self.rng.multivariate_normal(np.zeros(Omega.shape[0]), Omega, size=(self.mc_size,))
        noise_randomness_samples = self.rng.normal(size=(self.mc_size,))
        def compute_samples(local_fields, noise_randomness):
            network_local_field = local_fields[:self.p]
            target_local_field = local_fields[self.p:]
            return [f(network_local_field, target_local_field, noise_randomness) for f in fs]
        datas = [compute_samples(local_fields, noise_randomness) for local_fields, noise_randomness in tqdm(zip(local_fields_samples, noise_randomness_samples), total=self.mc_size)]
        datas = list(zip(*datas))
        datas = [np.array(data) for data in datas]
        if std:
            raise NotImplementedError
            return np.mean(data, axis=0), np.std(data, axis=0)/np.sqrt(self.mc_size)
        else:
            if not single_input:
                return (np.mean(data, axis=0) for data in datas)
            else:
                return np.mean(datas[0], axis=0)


    def compute_expected_values(self):
        def local_field_term_target(network_field, target_field, noise_randomness):
            return (
                (self.target(target_field)+np.sqrt(self.noise)*noise_randomness-self.network(network_field)) *
                np.einsum('j,r->jr', self.activation_derivative(network_field), target_field)
            )
        
        def local_field_term_network(network_field, target_field, noise_randomness):
            return (
                (self.target(target_field)+np.sqrt(self.noise)*noise_randomness-self.network(network_field)) *
                np.einsum('j,l->jl', self.activation_derivative(network_field), network_field)
            )
        
        def local_field_I4(network_field, target_field, noise_randomness):
            return (
                (self.target(target_field)+np.sqrt(self.noise)*noise_randomness-self.network(network_field))**2 *
                np.einsum('j,l->jl', self.activation_derivative(network_field), self.activation_derivative(network_field))
            )
        

        return self.local_fields_montecarlo((local_field_term_target, local_field_term_network, local_field_I4)) 

    def error(self):
        def local_field_error(network_field, target_field, noise_randomness):
            return .5*(self.target(target_field)+np.sqrt(self.noise)*noise_randomness-self.network(network_field))**2
        
        return self.local_fields_montecarlo(local_field_error)

