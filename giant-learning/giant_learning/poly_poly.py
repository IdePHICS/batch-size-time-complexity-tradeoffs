import numpy as np

from .base import SpecializedOverlapsBase, ProjectedOverlapsBase

class PolynomialPolynomialOverlapsBase(SpecializedOverlapsBase):
    def compute_expected_values(self):
        ev_target = np.zeros(shape=(self.p, self.k))
        ev_network = np.zeros(shape=(self.p, self.p))
        ev_I4 = np.zeros(shape=(self.p, self.p))

        # EV target
        for j in range(self.p):
            for r in range(self.k):
                for t in range(self.k):
                    ev_target[j, r] += 1/self.k * self._I3(
                        self.Q[j,j], self.M[j,r], self.M[j,t],
                        self.P[r,r], self.P[r,t],
                        self.P[t,t]
                    )
                if self.predictor_interaction:
                    for s in range(self.p):
                        ev_target[j, r] -= 1/self.p * self.a[s] * self._I3(
                            self.Q[j,j], self.M[j,r], self.Q[j,s],
                            self.P[r,r], self.M[s,r],
                            self.Q[s,s]
                        )
        # EV network
        for j in range(self.p):
            for l in range(self.p):
                for t in range(self.k):
                    ev_network[j, l] += 1/self.k * self._I3(
                        self.Q[j,j], self.Q[j,l], self.M[j,t],
                        self.Q[l,l], self.M[l,t],
                        self.P[t,t]
                    )
                if self.predictor_interaction:
                    for s in range(self.p):
                        ev_network[j, l] -= 1/self.p * self.a[s] * self._I3(
                            self.Q[j,j], self.Q[j,l], self.Q[j,s],
                            self.Q[l,l], self.Q[l,s],
                            self.Q[s,s]
                        )
        # EV I4
        for j in range(self.p):
            for l in range(self.p):
                ev_I4[j, l] += self.noise * self._I2noise(self.Q[j,j], self.Q[j,l], self.Q[l,l])

                for r in range(self.k):
                    for t in range(self.k):
                        ev_I4[j, l] += 1/self.k**2 * self._I4(
                            self.Q[j,j], self.Q[j,l], self.M[j,r], self.M[j,t],
                            self.Q[l,l], self.M[l,r], self.M[l,t],
                            self.P[r,r], self.P[r,t],
                            self.P[t,t]
                        )

                if self.predictor_interaction:
                    for s in range(self.p):
                        for u in range(self.p):
                            ev_I4[j, l] += 1/self.p**2 * self.a[s] * self.a[u] * self._I4(
                                self.Q[j,j], self.Q[j,l], self.Q[j,s], self.Q[j,u],
                                self.Q[l,l], self.Q[l,s], self.Q[l,u],
                                self.Q[s,s], self.Q[s,u],
                                self.Q[u,u]
                            )

                    for s in range(self.p):
                        for r in range(self.k):
                            ev_I4[j, l] -= 2/(self.p*self.k) * self.a[s] * self._I4(
                                self.Q[j,j], self.Q[j,l], self.Q[j,s], self.M[j,r],
                                self.Q[l,l], self.Q[l,s], self.M[l,r],
                                self.Q[s,s], self.M[s,r],
                                self.P[r,r]
                            )
            
        return ev_target, ev_network, ev_I4
    
    def error(self):
        population_error = self.noise
        for j in range(self.p):
            for l in range(self.p):
                population_error += 1/self.p**2 * self.a[j] * self.a[l] * self._I2(self.Q[j,j], self.Q[j,l], self.Q[l,l])
        for j in range(self.p):
            for r in range(self.k):
                population_error -= 2/(self.p*self.k) * self.a[j] * self._I2(self.Q[j,j], self.M[j,r], self.P[r,r])
        for r in range(self.k):
            for t in range(self.k):
                population_error += 1/self.k**2 * self._I2(self.P[r,r], self.P[r,t], self.P[t,t])
        return population_error/2.
        

class H2H2Overlaps(PolynomialPolynomialOverlapsBase):
    @staticmethod
    def _target(local_fields):
        k = len(local_fields) 
        return 1/k * np.sum(local_fields**2-1,axis=-1)
    
    @staticmethod
    def _activation(x):
        return x**2 - 1
    
    @staticmethod
    def _activation_derivative(x):
        return 2 * x

    @staticmethod
    def _I2noise(Caa,Cab,Cbb):
        return 4*Cab

    @staticmethod
    def _I2(Caa, Cab, Cbb):
        return 1. - Caa + 2*Cab**2 - Cbb + Caa*Cbb

    @staticmethod
    def _I3(Caa, Cab, Cac, Cbb, Cbc, Ccc):
        return -2*Cab + 4*Cac*Cbc + 2*Cab*Ccc

    @staticmethod
    def _I4(Caa, Cab, Cac, Cad, Cbb, Cbc, Cbd, Ccc, Ccd, Cdd):
        return 4*Cab - 8*Cac*Cbc - 8*Cad*Cbd - 4*Cab*Ccc + 8*Cad*Cbd*Ccc + 16*Cad*Cbc*Ccd + 16*Cac*Cbd*Ccd + 8*Cab*Ccd**2 - 4*Cab*Cdd + 8*Cac*Cbc*Cdd + 4*Cab*Ccc*Cdd
    
class H3H3Overlaps(PolynomialPolynomialOverlapsBase):
    @staticmethod
    def _target(local_fields):
        k = len(local_fields) 
        return 1/k * np.sum(local_fields**3-3*local_fields,axis=-1)
    
    @staticmethod
    def _activation(x):
        return x**3 - 3*x
    
    @staticmethod
    def _activation_derivative(x):
        return 3 * x**2 - 3

    @staticmethod
    def _I2noise(Caa,Cab,Cbb):
        return 9 - 9*Caa + 18*Cab**2 - 9*Cbb + 9*Caa*Cbb

    @staticmethod
    def _I2(Caa, Cab, Cbb):
        return 9*Cab - 9*Caa*Cab + 6*Cab**3 - 9*Cab*Cbb + 9*Caa*Cab*Cbb

    @staticmethod
    def _I3(Caa, Cab, Cac, Cbb, Cbc, Ccc):
        return -18*Cab*Cac + 9*Cbc - 9*Caa*Cbc + 18*Cac**2*Cbc + 18*Cab*Cac*Ccc - 9*Cbc*Ccc + 9*Caa*Cbc*Ccc

    @staticmethod
    def _I4(Caa, Cab, Cac, Cad, Cbb, Cbc, Cbd, Ccc, Ccd, Cdd):
        return (
            -162*Cac*Cad + 162*Cac*Cad*Cbb + 324*Cab*Cad*Cbc - 324*Cac*Cad*Cbc**2 + 324*Cab*Cac*Cbd - 162*Cbc*Cbd + 162*Caa*Cbc*Cbd - 
            324*Cac**2*Cbc*Cbd - 324*Cad**2*Cbc*Cbd - 324*Cac*Cad*Cbd**2 + 162*Cac*Cad*Ccc - 162*Cac*Cad*Cbb*Ccc - 324*Cab*Cad*Cbc*Ccc - 
            324*Cab*Cac*Cbd*Ccc + 162*Cbc*Cbd*Ccc - 162*Caa*Cbc*Cbd*Ccc + 324*Cad**2*Cbc*Cbd*Ccc + 324*Cac*Cad*Cbd**2*Ccc + 81*Ccd - 
            81*Caa*Ccd + 162*Cab**2*Ccd + 162*Cac**2*Ccd + 162*Cad**2*Ccd - 81*Cbb*Ccd + 81*Caa*Cbb*Ccd - 162*Cac**2*Cbb*Ccd - 
            162*Cad**2*Cbb*Ccd - 648*Cab*Cac*Cbc*Ccd + 162*Cbc**2*Ccd - 162*Caa*Cbc**2*Ccd + 324*Cad**2*Cbc**2*Ccd - 648*Cab*Cad*Cbd*Ccd + 
            1296*Cac*Cad*Cbc*Cbd*Ccd + 162*Cbd**2*Ccd - 162*Caa*Cbd**2*Ccd + 324*Cac**2*Cbd**2*Ccd - 81*Ccc*Ccd + 81*Caa*Ccc*Ccd - 
            162*Cab**2*Ccc*Ccd - 162*Cad**2*Ccc*Ccd + 81*Cbb*Ccc*Ccd - 81*Caa*Cbb*Ccc*Ccd + 162*Cad**2*Cbb*Ccc*Ccd + 
            648*Cab*Cad*Cbd*Ccc*Ccd - 162*Cbd**2*Ccc*Ccd + 162*Caa*Cbd**2*Ccc*Ccd - 324*Cac*Cad*Ccd**2 + 324*Cac*Cad*Cbb*Ccd**2 + 
            648*Cab*Cad*Cbc*Ccd**2 + 648*Cab*Cac*Cbd*Ccd**2 - 324*Cbc*Cbd*Ccd**2 + 324*Caa*Cbc*Cbd*Ccd**2 + 54*Ccd**3 - 54*Caa*Ccd**3 + 
            108*Cab**2*Ccd**3 - 54*Cbb*Ccd**3 + 54*Caa*Cbb*Ccd**3 + 162*Cac*Cad*Cdd - 162*Cac*Cad*Cbb*Cdd - 324*Cab*Cad*Cbc*Cdd + 
            324*Cac*Cad*Cbc**2*Cdd - 324*Cab*Cac*Cbd*Cdd + 162*Cbc*Cbd*Cdd - 162*Caa*Cbc*Cbd*Cdd + 324*Cac**2*Cbc*Cbd*Cdd - 
            162*Cac*Cad*Ccc*Cdd + 162*Cac*Cad*Cbb*Ccc*Cdd + 324*Cab*Cad*Cbc*Ccc*Cdd + 324*Cab*Cac*Cbd*Ccc*Cdd - 162*Cbc*Cbd*Ccc*Cdd + 
            162*Caa*Cbc*Cbd*Ccc*Cdd - 81*Ccd*Cdd + 81*Caa*Ccd*Cdd - 162*Cab**2*Ccd*Cdd - 162*Cac**2*Ccd*Cdd + 81*Cbb*Ccd*Cdd - 
            81*Caa*Cbb*Ccd*Cdd + 162*Cac**2*Cbb*Ccd*Cdd + 648*Cab*Cac*Cbc*Ccd*Cdd - 162*Cbc**2*Ccd*Cdd + 162*Caa*Cbc**2*Ccd*Cdd + 
            81*Ccc*Ccd*Cdd - 81*Caa*Ccc*Ccd*Cdd + 162*Cab**2*Ccc*Ccd*Cdd - 81*Cbb*Ccc*Ccd*Cdd + 81*Caa*Cbb*Ccc*Ccd*Cdd
        )
    
class H4H4Overlaps(PolynomialPolynomialOverlapsBase):
    @staticmethod
    def _target(local_fields):
        k = len(local_fields) 
        return 1/k * np.sum(local_fields**4-6*local_fields**2+3,axis=-1)
    
    @staticmethod
    def _activation(x):
        return x**4 - 6*x**2 + 3
    
    @staticmethod
    def _activation_derivative(x):
        return 4*x**3 - 12*x

    @staticmethod
    def _I2(Caa, Cab, Cbb):
        return (
            9 - 18*Caa + 9*Caa**2 + 72*Cab**2 - 72*Caa*Cab**2 + 24*Cab**4 - 18*Cbb + 36*Caa*Cbb - 18*Caa**2*Cbb - 72*Cab**2*Cbb + 
            72*Caa*Cab**2*Cbb + 9*Cbb**2 - 18*Caa*Cbb**2 + 9*Caa**2*Cbb**2
        )
    

class ProjectedH2H2Overlaps(ProjectedOverlapsBase, H2H2Overlaps):
    pass

class ProjectedH3H3Overlaps(ProjectedOverlapsBase, H3H3Overlaps):
    pass

class ProjectedH4H4Overlaps(ProjectedOverlapsBase, H4H4Overlaps):
    pass