#!/usr/bin/env python
from numpy.distutils.core import setup, Extension
from numpy.distutils.misc_util import Configuration

import distutils.sysconfig as ds

conf = Configuration()

setup(name='PEC',
      version='1.0',
      description='Tools for exoplanet transit light curve analysis.',
      author='Hannu Parviainen',
      author_email='hpparvia@gmail.com',
      url='',
      extra_options = ['-fopenmp'],
      package_dir={'pec':'src'},
      packages=['pec']
     )
