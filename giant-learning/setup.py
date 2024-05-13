import setuptools
from setuptools.extension import Extension

try:
    from Cython.Build import cythonize
except:
  print('You need Cython to install this package!')
  exit(1)
try:
  import numpy
except:
  print('You need numpy to install this package!')
  exit(1)

from giant_learning  import (
  __pkgname__ as PKG_NAME,
  __author__  as AUTHOR,
  __version__ as VERSION
)

def get_extensions():

  include_dirs = [
    numpy.get_include(),
    'external_libraries/boostmath/include'
  ]

  extra_compile_args = [
    '-O3',
    '-funroll-loops',
    '-std=c++17'
  ]

  cython_erf_erf = Extension(
      name='giant_learning.cython_erf_erf',
      sources=['giant_learning/erf_erf.pyx', 'giant_learning/erf_erf_integrals.cpp'],
      include_dirs=include_dirs,
      extra_compile_args=extra_compile_args
  )

  return cythonize(
    [cython_erf_erf],
    compiler_directives={'language_level':3},
    annotate=True
  )

setuptools.setup(
  setup_requires= [
    'Cython',
    'numpy',
    'setuptools>=18.0' 
  ],
  ext_modules=get_extensions(),
  name = PKG_NAME,
  author  =  AUTHOR,
  version = VERSION,
  packages = setuptools.find_packages(),
  python_requires = '>=3.7', # Probably it works even with newer version of python, but still...
  install_requires = [
    'numpy',
    'scipy',
  ],
  zip_safe=False
)
