import numpy as np

from .base import SpecializedOverlapsBase

# Constants
_pi = np.pi
_sqrtpi = np.sqrt(np.pi)
_pisqrtpi = np.pi * _sqrtpi
_pisquared = _pi**2

def _not_implemented_decorator(method):
    def wrapper(*args, **kwargs):
        raise NotImplementedError('StudentBase_Staircase2 is an abstract class, this methos is student specific')
    return wrapper


class StudentBase_Staircase2(SpecializedOverlapsBase):
    r"""
    Teacher is f^*(\vec{\lf}^*) = \lf^*_0 + \lf^*_0 * \lf^*_1

    Student is f(\vec{\lf}) = \frac1p \sum_{j=1}^p \sigma(\lf_j)

    Methods needed to work:
     - _activation
     - _activation_derivative
     - _actprime_lft0_target
     - _actprime_lft1_target
     - _actprime_lfs_target
     - _I3
     - _actprime_actprime_targetsquare
     - _actprime_actprime_target_act
     - _I4
     - _I2noise

     Mapping between indexes:
        - P[0,0] -> C11
        - P[0,1] -> C12
        - M[j,0] -> C13
        - M[l,0] -> C14
        - M[s,0] -> C15
        - M[u,0] -> C16
        - P[1,1] -> C22
        - M[j,1] -> C23
        - M[l,1] -> C24
        - M[s,1] -> C25
        - M[u,1] -> C26
        - Q[j,j] -> C33
        - Q[j,l] -> C34
        - Q[j,s] -> C35
        - Q[j,u] -> C36
        - Q[l,l] -> C44
        - Q[l,s] -> C45
        - Q[l,u] -> C46
        - Q[s,s] -> C55
        - Q[s,u] -> C56
        - Q[u,u] -> C66
    """
    @staticmethod
    def _target(local_fields):
        return local_fields[...,0] + local_fields[...,0] * local_fields[...,1]

    # def __init__(self, P: np.array, M0: np.array, Q0: np.array, a0: np.array, gamma: float, noise: float, I4_diagonal:bool, I4_offdiagonal:bool, second_layer_update: bool):
    #     # Check that k=2
    #     assert(P.shape[0] == 2)
    #     assert(M0.shape[1] == 2)
    #     super().__init__(self._target, self._activation, self._activation_derivative, P, M0, Q0, a0, gamma, noise, I4_diagonal, I4_offdiagonal, second_layer_update)

    def compute_expected_values(self):
        ev_target = np.zeros(shape=(self.p, self.k))
        ev_network = np.zeros(shape=(self.p, self.p))
        ev_I4 = np.zeros(shape=(self.p, self.p))

        # EV target
        for j in range(self.p):
            # target part
            ev_target[j, 0] += self._actprime_lft0_target(self.P[0,0], self.P[0,1], self.M[j,0], self.P[1,1], self.M[j,1], self.Q[j,j])
            ev_target[j, 1] += self._actprime_lft1_target(self.P[0,0], self.P[0,1], self.M[j,0], self.P[1,1], self.M[j,1], self.Q[j,j])
                
            # network part
            for s in range(self.p):
                ev_target[j, 0] -= 1./self.p * self.a[s] * self._I3(
                    self.Q[j,j], self.M[j,0], self.Q[j,s], self.P[0,0], self.M[s,0], self.Q[s,s]
                )
                ev_target[j, 1] -= 1./self.p * self.a[s] * self._I3(
                    self.Q[j,j], self.M[j,1], self.Q[j,s], self.P[1,1], self.M[s,1], self.Q[s,s]
                )
        
        # EV network
        for j in range(self.p):
            for l in range(self.p):
                # target part
                ev_network[j, l] += self._actprime_lfs_target(
                    self.P[0,0], self.P[0,1], self.M[j,0], self.M[l,0],
                    self.P[1,1], self.M[j,1], self.M[l,1], 
                    self.Q[j,j], self.Q[j,l], 
                    self.Q[l,l]
                    )
                
                # network part
                for s in range(self.p):
                    ev_network[j, l] -= 1./self.p * self.a[s] * self._I3(
                        self.Q[j,j], self.Q[j,l], self.Q[j,s], self.Q[l,l], self.Q[l,s], self.Q[s,s]
                    )
        
        # EV I4
        for j in range(self.p):
            for l in range(self.p):
                # target-target part
                ev_I4[j, l] += self._actprime_actprime_targetsquare(
                    self.P[0,0], self.P[0,1], self.M[j,0], self.M[l,0],
                    self.P[1,1], self.M[j,1], self.M[l,1],
                    self.Q[j,j], self.Q[j,l], 
                    self.Q[l,l]
                )

                for s in range(self.p):
                    # target-network part
                    ev_I4[j, l] -= 1./self.p * self.a[s] * self._actprime_actprime_target_act(
                        self.P[0,0], self.P[0,1], self.M[j,0], self.M[l,0], self.M[s,0],
                        self.P[1,1], self.M[j,1], self.M[l,1], self.M[s,1],
                        self.Q[j,j], self.Q[j,l], self.Q[j,s], 
                        self.Q[l,l], self.Q[l,s], 
                        self.Q[s,s]
                    )
                    # network-network part
                    for u in range(self.p):
                        ev_I4[j, l] += 1./self.p**2 * self.a[s] * self.a[u] * self._I4(
                            self.Q[j,j], self.Q[j,l], self.Q[j,s], self.Q[j,u],
                            self.Q[l,l], self.Q[l,s], self.Q[l,u],
                            self.Q[s,s], self.Q[s,u],
                            self.Q[u,u]
                        )
                
                # noise part
                ev_I4[j, l] += self.noise * self._I2noise(
                    self.Q[j,j], self.Q[j,l], self.Q[l,l]
                )
        return ev_target, ev_network, ev_I4


    @staticmethod
    @_not_implemented_decorator
    def _actprime_lft0_target(C11, C12, C13, C22, C23, C33):
        pass
    
    @staticmethod
    @_not_implemented_decorator
    def _actprime_lft1_target(C11, C12, C13, C22, C23, C33):
        pass
    
    @staticmethod
    @_not_implemented_decorator
    def _I3(C33, C34, C35, C44, C45, C55):
        pass
    
    @staticmethod
    @_not_implemented_decorator
    def _actprime_lfs_target(C11, C12, C13, C14, C22, C23, C24, C33, C34, C44):
        pass
    
    @staticmethod
    @_not_implemented_decorator
    def _actprime_actprime_targetsquare(C11, C12, C13, C14, C22, C23, C24, C33, C34, C44):
        pass
    
    @staticmethod
    @_not_implemented_decorator
    def _actprime_actprime_target_act(C11, C12, C13, C14, C15, C22, C23, C24, C25, C33, C34, C35, C44, C45, C55):
        pass
    
    @staticmethod
    @_not_implemented_decorator
    def _I4(C33, C34, C35, C36, C44, C45, C46, C55, C56, C66):
        pass
    
    @staticmethod
    @_not_implemented_decorator
    def _I2noise(C33, C34, C44):
        pass
    
    def error(self):
        error = self.noise/2.

        # target-target part
        error += self._target_target(self.P[0,0], self.P[0,1], self.P[1,1])

        for j in range(self.p):
            # target-network part
            error -= 2./self.p * self.a[j] * self._act_target(
                self.P[0,0], self.P[0,1], self.M[j,0], self.P[1,1], self.M[j,1], self.Q[j,j]
            )

            # network-network part
            for l in range(self.p):
                error += 1./self.p**2 * self.a[j] * self.a[l] * self._I2(
                    self.Q[j,j], self.Q[j,l], self.Q[l,l]
                )

        return error
    
    @staticmethod
    @_not_implemented_decorator
    def _target_target(C11, C12, C22):
        pass
    
    @staticmethod
    @_not_implemented_decorator
    def _act_target(C11, C12, C13, C22, C23, C33):
        pass
    
    @staticmethod
    @_not_implemented_decorator
    def _I2(C33, C34, C44):
        pass

class Hermite2Relu_Staricase2(StudentBase_Staircase2):
    r"""
    The studend activation \sigma is the truncated ReLU function,
    up to the second Hermite polynomial.
    """

    @staticmethod
    def _activation(x):
        return 1./(4*_sqrtpi) + x/2 + x**2/(2*_sqrtpi)
    
    @staticmethod
    def _activation_derivative(x):
        return 1./2 + x/_sqrtpi
    
    @staticmethod
    def _actprime_lft0_target(C11, C12, C13, C22, C23, C33):
        return C11/2. + (2*C12*C13)/_sqrtpi + (C11*C23)/_sqrtpi
                                                                              
    @staticmethod
    def _actprime_lft1_target(C11, C12, C13, C22, C23, C33):
        return C12/2. + (C13*C22)/_sqrtpi + (2*C12*C23)/_sqrtpi                                                               
                                                                                              
    @staticmethod
    def _actprime_lfs_target(C11, C12, C13, C14, C22, C23, C24, C33, C34, C44):
        return C14/2. + (C14*C23)/_sqrtpi + (C13*C24)/_sqrtpi + (C12*C34)/_sqrtpi

    @staticmethod
    def _I3(C33, C34, C35, C44, C45, C55): # Omming the arguments C1x, C2x since target is not involved
        return C34/(4.*_pi) + C45/4. + (C35*C45)/_pi + (C34*C55)/(2.*_pi)
                                                                                              
    @staticmethod
    def _actprime_actprime_targetsquare(C11, C12, C13, C14, C22, C23, C24, C33, C34, C44):
        return (
            C11/4. + C12**2/2. + (2*C12*C13)/_sqrtpi + (2*C12*C14)/_sqrtpi + (2*C13*C14)/_pi + (C11*C22)/4. + (2*C13*C14*C22)/_pi +
            (C11*C23)/_sqrtpi + (4*C12*C14*C23)/_pi + (C11*C24)/_sqrtpi + (4*C12*C13*C24)/_pi + (2*C11*C23*C24)/_pi + (C11*C34)/_pi +
            (2*C12**2*C34)/_pi + (C11*C22*C34)/_pi
        )
                                                                                              
    @staticmethod
    def _actprime_actprime_target_act(C11, C12, C13, C14, C15, C22, C23, C24, C25, C33, C34, C35, C44, C45, C55):
        return (
            C12/(16.*_sqrtpi) + C13/(8.*_pi) + C14/(8.*_pi) + C15/8. + (C14*C23)/(4.*_pisqrtpi) + (C15*C23)/(4.*_sqrtpi) +
            (C13*C24)/(4.*_pisqrtpi) + (C15*C24)/(4.*_sqrtpi) + (C13*C25)/(4.*_sqrtpi) + (C14*C25)/(4.*_sqrtpi) + (C15*C25)/(4.*_sqrtpi) + 
            (C12*C34)/(4.*_pisqrtpi) + (C15*C34)/(2.*_pi) + (C15*C25*C34)/_pisqrtpi + (C12*C35)/(4.*_sqrtpi) + (C14*C35)/(2.*_pi) + 
            (C15*C35)/(2.*_pi) + (C15*C24*C35)/_pisqrtpi + (C14*C25*C35)/_pisqrtpi + (C12*C45)/(4.*_sqrtpi) + (C13*C45)/(2.*_pi) + 
            (C15*C45)/(2.*_pi) + (C15*C23*C45)/_pisqrtpi + (C13*C25*C45)/_pisqrtpi + (C12*C35*C45)/_pisqrtpi + (C12*C55)/(8.*_sqrtpi) + 
            (C13*C55)/(4.*_pi) + (C14*C55)/(4.*_pi) + (C14*C23*C55)/(2.*_pisqrtpi) + (C13*C24*C55)/(2.*_pisqrtpi) + (C12*C34*C55)/(2.*_pisqrtpi)
        )
    
    @staticmethod
    def _I4(C33, C34, C35, C36, C44, C45, C46, C55, C56, C66):
        return (
            1/(64.*_pi) + C34/(16.*_pisquared) + C35/(16.*_pi) + C36/(16.*_pi) + C45/(16.*_pi) + (C35*C45)/(4.*_pisquared) + (C36*C45)/(4.*_pi) +
            C46/(16.*_pi) + (C35*C46)/(4.*_pi) + (C36*C46)/(4.*_pisquared) + C55/(32.*_pi) + (C34*C55)/(8.*_pisquared) + (C36*C55)/(8.*_pi) +
            (C46*C55)/(8.*_pi) + (C36*C46*C55)/(2.*_pisquared) + C55/16. + (C34*C55)/(4.*_pi) + (C35*C55)/(4.*_pi) + (C36*C55)/(4.*_pi) +
            (C45*C55)/(4.*_pi) + (C36*C45*C55)/_pisquared + (C46*C55)/(4.*_pi) + (C35*C46*C55)/_pisquared + C55**2/(8.*_pi) +
            (C34*C55**2)/(2.*_pisquared) + C66/(32.*_pi) + (C34*C66)/(8.*_pisquared) + (C35*C66)/(8.*_pi) + (C45*C66)/(8.*_pi) +
            (C35*C45*C66)/(2.*_pisquared) + (C55*C66)/(16.*_pi) + (C34*C55*C66)/(4.*_pisquared)
        )

    @staticmethod
    def _I2noise(C33, C34, C44):
        return 0.25 + C34/_pi
    
    @staticmethod
    def _I2(C33, C34, C44):
        return (
            1/(16.*_pi) + C33/(8.*_pi) + C34/4. + C34**2/(2.*_pi) + C44/(8.*_pi) + (C33*C44)/(4.*_pi)
        )
    
    @staticmethod
    def _target_target(C11, C12, C22):
        return (
            C11 + 2*C12**2 + C11*C22
        )
    
    @staticmethod
    def _act_target(C11, C12, C13, C22, C23, C33):
        return (
            C12/(4.*_sqrtpi) + C13/2. + (C13*C23)/_sqrtpi + (C12*C33)/(2.*_sqrtpi)
        )
    

