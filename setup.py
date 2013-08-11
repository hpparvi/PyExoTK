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
      url='https://github.com/hpparvi/PyExoChar',
      extra_options = ['-fopenmp'],
      package_dir={'exotk':'src'},
      packages=['exotk','exotk.utils']
     )
