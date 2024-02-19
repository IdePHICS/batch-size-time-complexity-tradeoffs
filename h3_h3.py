from giant_learning.gradient_descent import SphericalGradientDescent, GradientDescent
from giant_learning.montecarlo_overlaps import MonteCarloOverlaps
from giant_learning.poly_poly import SphericalH3H3Overlaps, H3H3Overlaps

import numpy as np
from scipy.special import erf
from sklearn.preprocessing import normalize
from scipy.linalg import orth
import matplotlib.pyplot as plt

p = 1
# l = 1.
k = 1
T = 117
noise = 1e-6
predictor_interaction = False
target = H3H3Overlaps._target
activation = H3H3Overlaps._activation
activation_derivative = H3H3Overlaps._activation_derivative
nseeds = 1
ds = np.logspace(10,10,base=2,num=1,dtype=int)

### save test error as a function of time for each seed and each d ###
simu_test_errors = np.zeros((nseeds, len(ds), T+1))
spherical_simu_test_errors = np.zeros((nseeds, len(ds), T+1))
theo_test_errors = np.zeros((nseeds, len(ds), T+1))
spherical_test_errors = np.zeros((nseeds, len(ds), T+1))
for i,d in enumerate(ds):
    n = int(np.power(d,1.3))
    t = 4/np.sqrt(d)  ### fix initial overlap
    gamma = .04 * n * np.power(d,-3/2)
    print(f'NOW Running d = {d}')
    for seed in range(nseeds):
        print(f'Seed = {seed}')
        rng = np.random.default_rng(seed+1)
        Wtarget = orth((normalize(rng.normal(size=(k,d)), axis=1, norm='l2')).T).T
        Wtild = normalize(rng.normal(size=(p,d)), axis=1, norm='l2')
        Wtild_target = np.einsum('ji,ri,rh->jh', Wtild , Wtarget ,Wtarget)
        W0_orth =  normalize(Wtild - Wtild_target, axis=1, norm='l2')
        W0 = -(t*normalize(Wtild_target,norm='l2',axis=1) + np.sqrt(1-t**2)*W0_orth)
        a0 = np.ones(p) ### It is changed with the new version of the package. The 1/p is included in giant-learning ###

        P = Wtarget @ Wtarget.T
        M0 = W0 @ Wtarget.T
        Q0 = W0 @ W0.T

        print(f'P = {P}')
        print(f'M0 = {M0}')
        print(f'Q0 = {Q0}')

        # Create a gradient descent object
        gd = SphericalGradientDescent(
            target, Wtarget, n,
            activation, W0, a0, activation_derivative,
            gamma, noise, predictor_interaction=not predictor_interaction,
            test_size = None, analytical_error= 'H3H3'
        )
        spherical_gd = SphericalGradientDescent(
            target, Wtarget, n,
            activation, W0, a0, activation_derivative,
            gamma, noise, predictor_interaction=predictor_interaction,
            test_size = None, analytical_error= 'H3H3'
        )

        gd.train(T)
        spherical_gd.train(T)

        print(f'GD test_error = {gd.test_errors[:2]}')
        print(f'spherical GD test_error = {spherical_gd.test_errors[:2]}')

        simu_test_errors[seed, i, :] = gd.test_errors
        spherical_simu_test_errors[seed, i, :] = spherical_gd.test_errors
        offdiag = (False if n == 1 else True)
        h3h3 = SphericalH3H3Overlaps(
            P, M0, Q0, a0,
            gamma, noise,
            I4_diagonal=d/n, I4_offdiagonal=offdiag,
            predictor_interaction=not predictor_interaction)
        
        h3h3_spherical = SphericalH3H3Overlaps(
            P, M0, Q0, a0,
            gamma, noise,
            I4_diagonal=d/n, I4_offdiagonal=offdiag,
            predictor_interaction=predictor_interaction)

        h3h3.train(T)
        h3h3_spherical.train(T)


        theo_test_errors[seed, i, :] = h3h3.test_errors
        spherical_test_errors[seed, i, :] = h3h3_spherical.test_errors


######## PLOTS ########

### Plot the average test error with std error bars as a function of time for different d ###
for i,d in enumerate(ds):
    # pass
    simu_plot = plt.errorbar(np.arange(T+1), np.mean(simu_test_errors[:,i,:], axis=0), yerr=np.std(simu_test_errors[:,i,:], axis=0), label=f'SGD Simulation', marker='x', ls='', color='red')
    spherical_simu_plot = plt.errorbar(np.arange(T+1), np.mean(spherical_simu_test_errors[:,i,:], axis=0), yerr=np.std(spherical_simu_test_errors[:,i,:], axis=0), label=f'Modified SGD Simulation', marker='x', ls='', color='blue')
theo_plot = plt.errorbar(np.arange(T+1), np.mean(theo_test_errors[:,i,:], axis=0), yerr=np.std(theo_test_errors[:,i,:], axis=0), label=f'SGD Theory', marker='', linestyle='-', color='red')
spherical_plot = plt.errorbar(np.arange(T+1), abs(np.mean(spherical_test_errors[:,i,:], axis=0)), yerr=np.std(spherical_test_errors[:,i,:], axis=0), label=f'Modified SGD Theory', marker='', linestyle='-', color='blue')
plt.xlabel('Steps')
plt.ylabel('Test error')
plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.show()

