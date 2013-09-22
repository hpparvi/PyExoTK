"""Astrophysical quantities

 Astrophysical quantities with their error estimates
 ===================================================

 Notes
 -----
  Error estimates follow the form Q_e where Q is the
  quantity in question, and are available only for a
  subset of quantities.

 =============  ============================== ======
 ``au``         astronomical unit                   m
 ``msun``       solar mass                         kg
 ``rsun``       solar radius                        m
 ``dsun``       mean solar mass density        g/cm^3
 ``mjup``       Jupiter mass                       kg
 ``rjup``       volumetric mean Jupiter radius      m
 ``rjup_eq``    equatorial Jupiter radius           m
 ``djup``       mean Jupiter mass density      g/cm^3
 ``mnep``       Neptune mass            radius     kg
 ``rnep``       volumetric mean Neptune radius      m
 ``rnep_eq``    equatorial Neptune radius           m
 ``djup``       mean Neptune mass density      g/cm^3
 ``rearth``     volumetric mean Earth radius        m
 ``rearth_eq``  equatorial Earth radius             m
 =============  ============================== ======
"""
from __future__ import division

from scipy.constants import G, pi

au,   au_e             = 1.496e11, 0.0       
msun, msun_e           = 1.9891e30, 0.0      
rsun, rsun_e           = 0.5*1.392684e9, 0.0
dsun, dsun_e           = 1.408, 0.0
mjup, mjup_e           = 1.89896e27, 0.0     
rjup, rjup_e           = 6.9911e7, 0.0       
rjup_eq, rjup_eq_e     = 7.1492e7, 0.0
djup, djup_e           = 1.326, 0.0       
mnep, mnep_e           = 1.0242e26, 0.0
rnep, mnep_e           = 2.4622e7, 0.0
rnep_eq, mnep_eq_e     = 2.4764e7, 0.0
dnep, dnep_e           = 1.638, 0.0       
mearth, mearth_e       = 5.9726e24, 0.0
rearth, rearth_e       = 6.371e6, 0.0
rearth_eq, rearth_eq_e = 6.3781e6, 0.0
dearth, dearth_e       = 5.514, 0.0       

def as_from_rhop(rho, period):
    """Calculates the scaled semi-major axis from the stellar density and planet's orbital period.

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
    """Calculates the semi-major axis from the stellar density, stellar radius, and planet's orbital period.

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
