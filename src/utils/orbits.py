from __future__ import division

import numpy as np
from scipy.constants import G, pi

def as_from_rhop(rho, period):
    """Scaled semi-major axis from the stellar density and planet's orbital period.

    Parameters
    ----------

      rho    : stellar density [g/cm^3]
      period : orbital period  [d]

    Returns
    -------

      as : scaled semi-major axis [R_star]
    """
    return (G/(3*pi))**(1/3)*((period*86400.)**2 * 1e3*rho)**(1/3)


def a_from_rhoprs(rho, period, rstar):
    """Semi-major axis from the stellar density, stellar radius, and planet's orbital period.

    Parameters
    ----------

      rho    : stellar density [g/cm^3]
      period : orbital period  [d]
      rstar  : stellar radius  [R_Sun]

    Returns
    -------

      a : semi-major axis [AU]
    """
    return as_from_rhop(rho,period)*rstar*rsun/au


def af_transit(e,w):
    """Calculates the -- factor during the transit"""
    return (1.0-e**2)/(1.0 + e*np.sin(w))


def i_from_baew(b,a,e,w):
    """Orbital inclination from the impact parameter, scaled semi-major axis, eccentricity and argument of periastron

    Parameters
    ----------

      b  : impact parameter       [-]
      a  : scaled semi-major axis [R_Star]
      e  : eccentricity           [-]
      w  : argument of periastron [rad]

    Returns
    -------

      i  : inclination            [rad]
    """
    return np.arccos(b / (a*af_transit(e, w)))
