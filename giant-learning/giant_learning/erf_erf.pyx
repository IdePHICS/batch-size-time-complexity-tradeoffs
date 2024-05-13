# distutils: language = c++
# distutils: sources = giant_learning/erf_erf_integrals.cpp

import numpy as np
cimport numpy as np
cimport cython
np.import_array() # Cython docs says I've always to use this with NumPy
from libc.math cimport sqrt, exp, erf, acos, asin

from ._config.python import DTYPE
# from ._config.cython cimport *
ctypedef np.float64_t DTYPE_t



cdef extern from 'erf_erf_integrals.cpp':
  cdef DTYPE_t square(DTYPE_t x)

cdef extern from 'erf_erf_integrals.cpp' namespace 'giant_learning::erf_erf':
  cdef inline DTYPE_t I2(DTYPE_t C11, DTYPE_t C12, DTYPE_t C22)
  cdef inline DTYPE_t I2_noise(DTYPE_t C11, DTYPE_t C12, DTYPE_t C22)
  cdef inline DTYPE_t I3(DTYPE_t C11, DTYPE_t C12, DTYPE_t C13, DTYPE_t C22, DTYPE_t C23, DTYPE_t C33)
  cdef inline DTYPE_t I4(DTYPE_t C11, DTYPE_t C12, DTYPE_t C13, DTYPE_t C14, DTYPE_t C22, DTYPE_t C23, DTYPE_t C24, DTYPE_t C33, DTYPE_t C34, DTYPE_t C44)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef erf_updates(np.ndarray[DTYPE_t, ndim=2] Q, np.ndarray[DTYPE_t, ndim=2] M, np.ndarray[DTYPE_t, ndim=2] P, np.ndarray[DTYPE_t, ndim=1] a, DTYPE_t noise):
  cdef np.ndarray[DTYPE_t, ndim=2] expected_I3_network = np.zeros_like(Q, dtype=DTYPE)
  cdef np.ndarray[DTYPE_t, ndim=2] expected_I3_target = np.zeros_like(M, dtype=DTYPE)
  cdef np.ndarray[DTYPE_t, ndim=2] expected_I4 = np.zeros_like(Q, dtype=DTYPE)
  cdef int p = Q.shape[0]
  cdef int k = P.shape[0]
  cdef DTYPE_t one_over_p = 1./p
  cdef DTYPE_t one_over_k = 1./k

  # Indexs for the for cycles (needed to have pure C for-loops)
  cdef int j,l,r,o,q,m,s

  ## Compute expected_I3_target
  # Fixed the pair (j,r)
  for j in range(p):
    for r in range(0,k):

      # Student
      for l in range(0,p):
        expected_I3_target[j,r] -= one_over_p * a[l] * I3(Q[j,j], M[j,r], Q[j,l], P[r,r], M[l,r], Q[l,l])

      # Teacher
      for s in range(0,k):
        expected_I3_target[j, r] += one_over_k * I3(Q[j,j], M[j,r], M[j,s], P[r,r], P[r,s], P[s,s]) 

  ### Compute expected_I3_network & expected_I4 
  for j in range(0,p):
    for l in range(0,p):

      ## Student
      for m in range(0,p):
        expected_I3_network[j,l] -= one_over_p * a[m] * I3(Q[j,j], Q[j,l], Q[j,m], Q[l,l], Q[l,m], Q[m,m])

      for r in range(0,k):
        expected_I3_network[j,l] += one_over_k * I3(Q[j,j], Q[j,l], M[j,r], Q[l,l], M[l,r], P[r,r]) 

      ## Noise term
      expected_I4[j,l] += noise * I2_noise(Q[j,j], Q[j,l], Q[l,l])
      
      ## I4 contribution
      # Student-student
      for o in range(0,p):
        for q in range(0,p):
          expected_I4[j,l] += square(one_over_p) * a[o]*a[q] * I4(Q[j,j], Q[j,l], Q[j,o], Q[j,q], Q[l,l], Q[l,o], Q[l,q], Q[o,o], Q[o,q], Q[q,q])

      # Student-teacher
      for o in range(0,p):
        for r in range(0,k):
          expected_I4[j,l] -= 2 * one_over_p * one_over_k * a[o] * I4(Q[j,j], Q[j,l], Q[j,o], M[j,r], Q[l,l], Q[l,o], M[l,r], Q[o,o], M[o,r], P[r,r])

      # Teacher-Teacher
      for r in range(0,k):
        for s in range(0,k):
          expected_I4[j,l] += square(one_over_k) * I4(Q[j,j], Q[j,l], M[j,r], M[j,s], Q[l,l], M[l,r], M[l,s], P[r,r], P[r,s], P[s,s])

  return expected_I3_target, expected_I3_network, expected_I4

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef erf_error(np.ndarray[DTYPE_t, ndim=2] Q, np.ndarray[DTYPE_t, ndim=2] M, np.ndarray[DTYPE_t, ndim=2] P, np.ndarray[DTYPE_t, ndim=1] a, DTYPE_t noise):
  cdef DTYPE_t risk = 0.

  cdef int p = Q.shape[0]
  cdef int k = P.shape[0]
  cdef DTYPE_t one_over_p = 1./p
  cdef DTYPE_t one_over_k = 1./k

  cdef int j,l

  # Teacher-Teacher
  for j in range(0,k):
    for l in range(0,k):
      risk += square(one_over_k) * I2(P[j,j], P[j,l], P[l,l])
  # Teacher-Student
  for j in range(0,p):
    for l in range(0,k):
      risk -= 2*one_over_p*one_over_k * a[j] * I2(Q[j,j], M[j,l], P[l,l])
  # Student-Student
  for j in range(0,p):
    for l in range(0,p):
      risk += square(one_over_p) * a[j]*a[l] * I2(Q[j,j], Q[j,l], Q[l,l])

  # Noise term
  risk += noise

  return risk/2


