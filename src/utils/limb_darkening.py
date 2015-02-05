import numpy as np
from scipy.optimize import fmin

def quadratic_law(mu,ld):
    """Quadratic limb-darkening law as  described in (Mandel & Agol, 2001).
    """
    ld = np.asarray(ld)
    if ld.ndim == 1:
        return 1. - ld[0]*(1.-mu) - ld[1]*(1.-mu)**2
    else:
        return 1. - ld[:,0,np.newaxis]*(1.-mu) - ld[:,1,np.newaxis]*(1.-mu)**2


def nonlinear_law(mu,ld):
    """Nonlinear limb darkening law as described in (Mandel & Agol, 2001).
    """
    return 1. - np.sum([ld[i-1]*(1.-mu**(0.5*i)) for i in range(1,5)], axis=0)


def general_law(mu,ld):
    """General limb darkening law as described in (Gimenez, 2006).
    """
    ld = np.asarray(ld)
    if ld.ndim == 1:
        return 1. - (ld[0]*(1.-mu) + ld[1]*(1.-mu**2) + ld[2]*(1.-mu**3) + ld[3]*(1.-mu**4)) 
    else:
        return 1. - (ld[:,0,np.newaxis]*(1.-mu) + ld[:,1,np.newaxis]*(1.-mu**2) + ld[:,2,np.newaxis]*(1.-mu**3) + ld[:,3,np.newaxis]*(1.-mu**4)) 


def nonlinear_to_general(ldc):
    """Estimate the general limb darkening law coefficients from the nonlinear law coefficients.
    """
    mu = np.linspace(0,1,300)
    I_nl = nonlinear_law(mu,ldc)
    return fmin(lambda pv:sum((I_nl-general_law(mu,pv))**2), ldc, disp=0)
    
