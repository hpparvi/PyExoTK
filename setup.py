#!/usr/bin/env python
from numpy.distutils.core import setup, Extension
from numpy.distutils.misc_util import Configuration

import distutils.sysconfig as ds

conf = Configuration()

setup(name='PyExoTK',
      version='0.5',
      description='Tools for exoplanet transit light curve analysis.',
      author='Hannu Parviainen',
      author_email='hpparvi@gmail.com',
      url='https://github.com/hpparvi/PyExoTK',
      extra_options = ['-fopenmp'],
      package_dir={'exotk':'src'},
      packages=['exotk','exotk.utils','exotk.lpf','exotk.io'],
      ext_modules=[ Extension('exotk.utils.misc_f', ['src/utils/utilities.f90'], libraries=['gomp']),
                    Extension('exotk.utils.likelihood_f', ['src/utils/likelihood.f90'], libraries=['gomp'])]
     )
